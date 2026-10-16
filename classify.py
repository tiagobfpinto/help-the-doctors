"""Select using labelled row development; report unseen-note robustness separately.

The official test is opened only after selection, fitting on all originals and
building the original-training exclusion index. Scores are exploratory
development estimates, not independent post-selection or official-test results.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from augmentation import LR_C, fit_evaluated_model
from augmentation_policy import (
    MODEL_IDS, MODEL_NAMES, DECODERS, PRIMARY_PROTOCOL, GROUPED_PROTOCOL,
    CANDIDATE_IDS, SOURCE_FILES, PLAN_FILENAME, POLICY_SPEC, SELECTION_RULE,
    INCUMBENT, PROMOTION_BASELINES, choose_candidate, validate_decision_plan,
)
from config import (
    FINAL_EVALUATION_DIR, FINAL_OUTPUT_DIR, MIN_ACCURACY, MIN_CLASS_F1,
    N_FOLDS, RESULTS_FILE, ROOT, SEED, SELECTED_MODEL_FILE, TEST_FILE, TRAIN_FILE,
)
from data_preparation import repair_training_csv, sha256
from duplicate_exclusion import fit_exclusion_index, score_model
from read_test_set import read_test_cases, write_results

IMPLEMENTATION_FILES = (
    "augmentation.py", "models.py", "data_preparation.py", "duplicate_exclusion.py", "augmentation_policy.py",
)
INFERENCE_FILES = ("classify.py", "read_test_set.py", "config.py")
RELEASE_FILES = (*IMPLEMENTATION_FILES, *INFERENCE_FILES, PLAN_FILENAME)
EVALUATION_STATUS = (
    "Exploratory development selection after corpus/annotation inspection and "
    "external methodological review; neither row nor grouped CV is an "
    "independent post-selection test evaluation. Exclusion assumes that an "
    "identical input's next annotation tends to use a previously unseen label."
)


def record_selection() -> dict:
    """Select from a completed train-only experiment, verifying its provenance."""
    summary_path = FINAL_EVALUATION_DIR / "summary.csv"
    experiment_path = FINAL_EVALUATION_DIR / "experiment.json"
    run_state_path = FINAL_EVALUATION_DIR / "run_state.json"
    class_path = FINAL_EVALUATION_DIR / "metrics_per_class.csv"
    if not all(path.is_file() for path in (summary_path, class_path, experiment_path, run_state_path)):
        raise FileNotFoundError("Run run_final_evaluation.py before selecting the final model.")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    state = json.loads(run_state_path.read_text(encoding="utf-8"))
    if experiment.get("state") != "complete" or state.get("state") != "complete":
        raise ValueError("The final evaluation is still running or incomplete.")
    if experiment.get("seed") != SEED or experiment.get("LR_C") != LR_C or experiment.get("n_folds") != N_FOLDS:
        raise ValueError("Current settings differ from the evaluated experiment.")
    validate_decision_plan(ROOT / PLAN_FILENAME, ROOT)
    sources = experiment.get("sources_sha256", {})
    if sources.get("train.csv") != sha256(TRAIN_FILE):
        raise ValueError("Development scores do not match the current training file.")
    for name in (*SOURCE_FILES[1:], PLAN_FILENAME):
        if sources.get(name) != sha256(ROOT / name):
            raise ValueError(f"{name} changed since evaluation; rerun the final evaluation.")
    artifact_hashes = experiment.get("artifacts_sha256", {})
    required_artifacts = {"summary.csv", "metrics_per_class.csv", "gate_diagnostics.csv",
                          "promotion_decision.json", "oof_predictions.csv", "design.json", PLAN_FILENAME}
    if not required_artifacts <= set(artifact_hashes):
        raise ValueError("The completed evaluation manifest is missing required evidence.")
    for name, expected in artifact_hashes.items():
        path = (FINAL_EVALUATION_DIR / name).resolve()
        if not path.is_relative_to(FINAL_EVALUATION_DIR.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Evaluation artifact changed or missing: {name}")
    selected = choose_candidate(pd.read_csv(summary_path, sep=";"), pd.read_csv(class_path, sep=";"))
    saved_decision = experiment.get("promotion_decision", {})
    if (saved_decision.get("candidate_id") != selected["candidate_id"]
            or saved_decision.get("promotion_applied") != selected["promotion_applied"]):
        raise ValueError("The persisted decision differs from the frozen promotion policy.")
    selected.update({
        "schema_version": 3, "selection_rule": SELECTION_RULE,
        "evaluation_status": EVALUATION_STATUS, "train_sha256": sources["train.csv"],
        "experiment_directory": FINAL_EVALUATION_DIR.relative_to(ROOT).as_posix(),
        "summary_sha256": sha256(summary_path), "experiment_sha256": sha256(experiment_path),
        "implementation_sha256": {name: sha256(ROOT / name) for name in IMPLEMENTATION_FILES},
        "inference_implementation_sha256": {name: sha256(ROOT / name) for name in INFERENCE_FILES},
        "release_files_sha256": {name: sha256(ROOT / name) for name in RELEASE_FILES},
        "decision_plan_sha256": sha256(ROOT / PLAN_FILENAME),
        "release_requirements": "Normal inference requires train.csv, official inputs, this selection, and the declared code/plan files; development artifacts are needed only for --select.",
        "LR_C": LR_C, "seed": SEED, "n_folds": N_FOLDS,
    })
    SELECTED_MODEL_FILE.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected


def load_selection() -> dict:
    selected = json.loads(SELECTED_MODEL_FILE.read_text(encoding="utf-8"))
    if selected.get("schema_version") != 3 or selected.get("candidate_id") not in {INCUMBENT, *PROMOTION_BASELINES}:
        raise ValueError("Legacy or unsupported selection; run the final evaluation and --select.")
    expected_id = selected.get("model_id", "") + ("+EXCL" if selected.get("decoder") == "exclude_seen" else "")
    if (selected.get("decoder") not in DECODERS or selected["candidate_id"] != expected_id
            or selected.get("model") != MODEL_NAMES.get(selected.get("model_id"))):
        raise ValueError("Selection contains an inconsistent classifier/decoder.")
    if selected.get("train_sha256") != sha256(TRAIN_FILE):
        raise ValueError("Selection is incompatible with the current training file.")
    if selected.get("LR_C") != LR_C or selected.get("seed") != SEED or selected.get("n_folds") != N_FOLDS:
        raise ValueError("Settings changed; rerun the final evaluation and selection.")
    if selected.get("protocol") != PRIMARY_PROTOCOL or selected.get("grouped_protocol") != GROUPED_PROTOCOL:
        raise ValueError("Selection uses an unsupported evaluation protocol.")
    if selected.get("selection_policy") != POLICY_SPEC:
        raise ValueError("Selection uses a different augmentation promotion policy.")
    diagnostics = selected.get("promotion_diagnostics", [])
    if {item.get("candidate_id") for item in diagnostics} != set(PROMOTION_BASELINES):
        raise ValueError("Selection omits the augmentation promotion diagnostics.")
    passing = {item["candidate_id"] for item in diagnostics if item.get("promotion_pass") is True}
    if selected.get("promotion_applied") != bool(passing):
        raise ValueError("Selection promotion flag differs from its diagnostics.")
    if ((not passing and selected["candidate_id"] != INCUMBENT)
            or (passing and selected["candidate_id"] not in passing)):
        raise ValueError("Selection violates the frozen promotion/fallback gate.")
    for field in ("development_metrics", "grouped_metrics"):
        metrics = selected.get(field, {})
        if set(metrics) != {"accuracy", "f1_macro", "min_f1", "f1_dermatology"}:
            raise ValueError("Selection is missing scenario metrics.")
        values = np.asarray(list(metrics.values()), dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Selection metrics are invalid.")
    required_hashes = {"implementation_sha256": set(IMPLEMENTATION_FILES),
                       "inference_implementation_sha256": set(INFERENCE_FILES),
                       "release_files_sha256": set(RELEASE_FILES)}
    for field, names in required_hashes.items():
        if set(selected.get(field, {})) != names:
            raise ValueError("Selection is missing implementation provenance; regenerate it.")
        for name, expected in selected[field].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"{name} changed since selection; regenerate selection as appropriate.")
    return selected


def predict_selected(fitted, exclusion_index, test: pd.DataFrame, selected: dict):
    """The same score/decoder helpers used in CV, with no inference targets."""
    scores, classes = score_model(fitted, test, selected["model_id"])
    result = exclusion_index.predict(scores, classes, test)
    predictions = result.predictions if selected["decoder"] == "exclude_seen" else result.raw_predictions
    return [str(value) for value in predictions], result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true", help="select from both scenarios using row development only")
    args = parser.parse_args(argv)
    selected = record_selection() if args.select or not SELECTED_MODEL_FILE.exists() else load_selection()
    records, _ = repair_training_csv(TRAIN_FILE)
    train = pd.DataFrame(records)
    with threadpool_limits(limits=1):
        fitted, batch = fit_evaluated_model(train, selected["model_id"], seed=SEED, fold=0)
    # Classifier and original-training index are fixed before official inputs.
    exclusion_index = fit_exclusion_index(train)
    test_cases = read_test_cases(TEST_FILE)
    predicted, result = predict_selected(fitted, exclusion_index, pd.DataFrame(test_cases), selected)
    write_results(RESULTS_FILE, predicted, test_cases)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics = result.diagnostics.copy()
    diagnostics.insert(0, "case_id", [case["case_id"] for case in test_cases])
    diagnostics["raw_prediction"] = result.raw_predictions
    diagnostics["predicted_class"] = predicted
    diagnostics_path = FINAL_OUTPUT_DIR / "decoding_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, sep=";", encoding="utf-8-sig", index=False)
    final_metadata = {
        "selected_model": selected, "original_training_cases": len(train),
        "fit_rows_including_descendants": len(batch.y), "training_batch": batch.metadata,
        "decoder": selected["decoder"], "exclusion_index_original_training_rows": exclusion_index.original_training_rows,
        "exclusion_key": "lowercase whitespace-normalized (sample_name, transcription); skip only wholly empty input",
        "exclusion_fallback": "keep raw prediction for no match or all classes excluded",
        "test_source_sha256": sha256(TEST_FILE), "test_cases": len(test_cases),
        "results_sha256": sha256(RESULTS_FILE), "results_lines": len(predicted),
        "diagnostics_sha256": sha256(diagnostics_path),
        "test_metrics": None, "test_metrics_reason": "No official test labels are available.",
        "versions": {name: metadata.version(name) for name in ("pandas", "numpy", "scikit-learn")},
    }
    (FINAL_OUTPUT_DIR / "inference.json").write_text(
        json.dumps(final_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Selected {selected['candidate_id']}: {selected['model']} / {selected['decoder']}")
    print(f"Fit on all {len(train)} original cases; wrote {len(predicted)} predictions to {RESULTS_FILE}.")
    print("Row/grouped scores are exploratory development CV. Official test scores are unknown.")


if __name__ == "__main__":
    main()
