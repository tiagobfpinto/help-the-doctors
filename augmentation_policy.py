"""Frozen development-only promotion policy for section recombination.

The policy follows earlier corpus/model audits. It is fixed before this new
experiment, but does not turn reused labelled development data into a test set.
Only augmented candidates can replace the incumbent; copies are controls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation import sha256

MODEL_IDS = ("NB", "B", "D", "E", "SVC", "SVC_AUG", "SVC_COPY")
MODEL_NAMES = {mid: "MultinomialNB" if mid == "NB" else "LinearSVC" if mid.startswith("SVC")
               else "LogisticRegression" for mid in MODEL_IDS}
DECODERS = ("raw", "exclude_seen")
PRIMARY_PROTOCOL = "stratified_rows_development"
GROUPED_PROTOCOL = "grouped_development"
PROTOCOLS = (PRIMARY_PROTOCOL, GROUPED_PROTOCOL)
CANDIDATE_IDS = {mid + ("+EXCL" if dec == "exclude_seen" else "")
                 for mid in MODEL_IDS for dec in DECODERS}
INCUMBENT = "SVC+EXCL"
PROMOTION_BASELINES = {"D+EXCL": "B+EXCL", "SVC_AUG+EXCL": INCUMBENT}
COPY_CONTROLS = {"D+EXCL": "E+EXCL", "SVC_AUG+EXCL": "SVC_COPY+EXCL"}
EPSILON = 1e-12
PLAN_FILENAME = "augmentation_decision_plan.json"
SOURCE_FILES = (
    "train.csv", "augmentation.py", "models.py", "data_preparation.py",
    "duplicate_exclusion.py", "augmentation_policy.py", "run_final_evaluation.py", "config.py",
)
POLICY_SPEC = {
    "version": 1,
    "primary_protocol": PRIMARY_PROTOCOL,
    "robustness_protocol": GROUPED_PROTOCOL,
    "seed": 0, "n_folds": 5, "LR_C": 0.25,
    "model_ids": list(MODEL_IDS), "decoders": list(DECODERS),
    "incumbent": INCUMBENT,
    "eligible_for_promotion": list(PROMOTION_BASELINES),
    "family_baselines": PROMOTION_BASELINES,
    "copy_controls": COPY_CONTROLS,
    "strict_improvement_epsilon": EPSILON,
    "required_improvements": ["f1_macro", "f1_dermatology"],
    "maximum_accuracy_drop": 0.01,
    "absolute_accuracy_strictly_greater_than": 0.49,
    "absolute_minimum_class_f1": 0.25,
    "must_pass_against": ["own_family_original", "incumbent"],
    "passing_candidates_order": ["f1_macro_desc", "f1_dermatology_desc", "accuracy_desc", "candidate_id_asc"],
    "if_none_pass": INCUMBENT,
    "copy_control_role": "diagnostic only; never eligible for promotion",
    "grouped_role": "separate robustness scenario; never used for promotion",
    "evaluation_status": "Fixed before the new run after prior development studies; exploratory, not independent test evaluation.",
}
SELECTION_RULE = (
    "Only D+EXCL and SVC_AUG+EXCL may replace SVC+EXCL. In seed-0 five-fold "
    "stratified-row development, each must improve both macro F1 and Dermatology "
    "F1 by more than 1e-12, with accuracy loss at most 0.01, against its own "
    "unaugmented family and against SVC+EXCL. Accuracy must exceed 0.49 and all "
    "class F1 scores must be at least 0.25. Rank passing candidates by macro F1, "
    "Dermatology F1, accuracy, then alphabetical ID. Otherwise retain SVC+EXCL. "
    "Matched copies and grouped evaluation are diagnostics, never promotion gates."
)


def validate_tables(summary: pd.DataFrame, per_class: pd.DataFrame) -> None:
    required = {"candidate_id", "model_id", "decoder", "model", "protocol", "seed",
                "accuracy", "f1_macro", "min_f1"}
    if missing := required - set(summary):
        raise ValueError(f"Missing experiment columns: {sorted(missing)}")
    if len(summary) != 28 or set(summary.protocol) != set(PROTOCOLS) or set(summary.seed) != {0}:
        raise ValueError("Require all 28 candidate/scenario rows at seed 0.")
    if summary.duplicated(["protocol", "candidate_id"]).any():
        raise ValueError("Duplicate candidate/scenario metrics.")
    for protocol in PROTOCOLS:
        if set(summary.loc[summary.protocol.eq(protocol), "candidate_id"]) != CANDIDATE_IDS:
            raise ValueError("Missing one of the fourteen declared candidates.")
    for row in summary.itertuples():
        expected = row.model_id + ("+EXCL" if row.decoder == "exclude_seen" else "")
        if (row.model_id not in MODEL_IDS or row.decoder not in DECODERS
                or row.candidate_id != expected or row.model != MODEL_NAMES[row.model_id]):
            raise ValueError("Inconsistent candidate ID, classifier or decoder.")
    values = summary[["accuracy", "f1_macro", "min_f1"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Metrics must be finite fractions in [0,1].")
    required_class = {"protocol", "seed", "candidate_id", "class", "f1"}
    if not required_class <= set(per_class):
        raise ValueError("Missing per-class metrics required by the Dermatology gate.")
    if set(per_class.protocol) != set(PROTOCOLS) or set(per_class.seed) != {0}:
        raise ValueError("Per-class protocols/seed differ from the declared experiment.")
    if per_class.duplicated(["protocol", "candidate_id", "class"]).any():
        raise ValueError("Duplicate per-class metric.")
    expected_classes = set(per_class["class"])
    if len(expected_classes) != 12 or "Dermatology" not in expected_classes:
        raise ValueError("Require all twelve classes including Dermatology.")
    if len(per_class) != 28 * 12:
        raise ValueError("Require twelve class metrics for each candidate/scenario.")
    for row in summary.itertuples():
        current = per_class.loc[per_class.protocol.eq(row.protocol) & per_class.candidate_id.eq(row.candidate_id)]
        if set(current["class"]) != expected_classes:
            raise ValueError("Incomplete or inconsistent per-class coverage.")
        f1 = current.f1.to_numpy(dtype=float)
        if not np.isfinite(f1).all() or ((f1 < 0) | (f1 > 1)).any():
            raise ValueError("Per-class F1 must be finite fractions.")
        if abs(float(f1.min()) - row.min_f1) > 1e-10 or abs(float(f1.mean()) - row.f1_macro) > 1e-10:
            raise ValueError("Summary F1 differs from the per-class report.")


def choose_candidate(summary: pd.DataFrame, per_class: pd.DataFrame) -> dict:
    """Apply the fixed gate and return a self-contained selection explanation."""
    validate_tables(summary, per_class)
    lookup = summary.set_index(["protocol", "candidate_id"])
    derm = per_class.loc[per_class["class"].eq("Dermatology")].set_index(["protocol", "candidate_id"]).f1

    def values(cid, protocol=PRIMARY_PROTOCOL):
        row = lookup.loc[(protocol, cid)]
        return {**{k: float(row[k]) for k in ("accuracy", "f1_macro", "min_f1")},
                "f1_dermatology": float(derm.loc[(protocol, cid)])}

    diagnostics, passing = [], []
    for cid, baseline in PROMOTION_BASELINES.items():
        score = values(cid)
        tests = {}
        deltas = {}
        for role, reference in (("family", baseline), ("incumbent", INCUMBENT)):
            reference_score = values(reference)
            delta = {metric: score[metric] - reference_score[metric]
                     for metric in ("accuracy", "f1_macro", "f1_dermatology")}
            deltas.update({f"{role}_delta_{key}": val for key, val in delta.items()})
            tests[f"{role}_macro_improved"] = delta["f1_macro"] > EPSILON
            tests[f"{role}_dermatology_improved"] = delta["f1_dermatology"] > EPSILON
            # Tiny numerical tolerance only protects the inclusive 1pp boundary.
            tests[f"{role}_accuracy_within_drop"] = delta["accuracy"] >= -0.01 - EPSILON
        tests["absolute_accuracy_pass"] = score["accuracy"] > 0.49
        tests["absolute_min_f1_pass"] = score["min_f1"] >= 0.25
        passed = all(tests.values())
        copy = values(COPY_CONTROLS[cid])
        diagnostics.append({
            "candidate_id": cid, "family_baseline": baseline, "incumbent": INCUMBENT,
            "copy_control": COPY_CONTROLS[cid], "protocol": PRIMARY_PROTOCOL,
            **score, **tests, **deltas,
            **{f"copy_delta_{k}": score[k] - copy[k] for k in ("accuracy", "f1_macro", "f1_dermatology")},
            "promotion_pass": passed,
            "failed_checks": ",".join(key for key, passed_check in tests.items() if not passed_check),
        })
        if passed:
            passing.append(cid)
    selected_id = min(passing, key=lambda cid: (-values(cid)["f1_macro"], -values(cid)["f1_dermatology"],
                                               -values(cid)["accuracy"], cid)) if passing else INCUMBENT
    row = lookup.loc[(PRIMARY_PROTOCOL, selected_id)]
    selected_metrics = values(selected_id)
    return {
        "candidate_id": selected_id, "model_id": str(row.model_id), "decoder": str(row.decoder),
        "model": str(row.model), "protocol": PRIMARY_PROTOCOL,
        "development_metrics": selected_metrics, "grouped_protocol": GROUPED_PROTOCOL,
        "grouped_metrics": values(selected_id, GROUPED_PROTOCOL),
        "meets_development_thresholds": selected_metrics["accuracy"] > .49 and selected_metrics["min_f1"] >= .25,
        "promotion_applied": bool(passing), "incumbent": INCUMBENT,
        "selection_policy": POLICY_SPEC, "selection_rule": SELECTION_RULE,
        "promotion_diagnostics": diagnostics,
    }


def validate_decision_plan(path: Path, root: Path) -> dict:
    """Refuse fitting unless review and sources were frozen before the run."""
    if not path.is_file():
        raise ValueError("No frozen augmentation decision plan; fitting is not authorized yet.")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen" or plan.get("approved_to_fit") is not True:
        raise ValueError("The decision plan is not frozen/approved for fitting.")
    if plan.get("policy_spec") != POLICY_SPEC:
        raise ValueError("The frozen decision policy differs from augmentation_policy.py.")
    review = plan.get("review", {})
    if review.get("reviewed_count", 0) < 5 or not review.get("reviewer"):
        raise ValueError("Five augmentation examples must be manually reviewed before fits.")
    examples_path = (root / review.get("examples_file", "")).resolve()
    if not examples_path.is_relative_to(root.resolve()) or not examples_path.is_file():
        raise ValueError("The reviewed examples artifact is missing or outside the project.")
    if review.get("examples_sha256") != sha256(examples_path):
        raise ValueError("The examples differ from the reviewed augmentation artifact.")
    expected_settings = {"seed": 0, "n_folds": 5, "LR_C": .25, "model_ids": list(MODEL_IDS), "protocols": list(PROTOCOLS)}
    if plan.get("settings") != expected_settings:
        raise ValueError("Frozen plan settings differ from the new experiment.")
    hashes = plan.get("sources_sha256", {})
    if set(hashes) != set(SOURCE_FILES):
        raise ValueError("Frozen plan must cover all declared training/evaluation sources.")
    for name in SOURCE_FILES:
        if hashes[name] != sha256(root / name):
            raise ValueError(f"{name} differs from the frozen decision plan.")
    return plan
