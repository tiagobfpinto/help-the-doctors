r"""Train-only independent audit of a duplicate-based class exclusion heuristic.

    .venv\Scripts\python.exe audit_duplicate_exclusion.py

No official test, final selected model, or production predictions are read or
written. The heuristic learns exact-note label sets only from each fold's
original training rows, then excludes those classes when ranking that fold's
validation case. It assumes a repeated note will receive a different category;
the audit measures when this dataset-specific assumption holds or fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from augmentation import (
    LABEL, LR_C, build_condition_model, build_evaluated_model, clean_inputs,
    identity_groups, normalize_text, prediction_input, prepare_training,
)
from data_preparation import repair_training_csv, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "auditoria_exclusao"
MODEL_IDS = ("NB", "B", "SVC")
EXPERIMENTS = (("stratified", 0), ("stratified", 1), ("stratified", 2), ("grouped", 0))
N_FOLDS = 5


def canonical_note(value) -> str:
    return normalize_text(value).lower()


def training_note_labels(notes, labels) -> dict[str, set[str]]:
    """No validation labels are accepted or accessed by this function."""
    result: dict[str, set[str]] = {}
    if len(notes) != len(labels):
        raise ValueError("Notes and training labels must have the same length.")
    for note, label in zip(notes, labels):
        key = canonical_note(note)
        if key:
            result.setdefault(key, set()).add(str(label))
    return result


def exclude_seen_classes(scores, classes, validation_notes, seen_labels):
    """Rank classes after excluding ALL fold-training labels of a nonempty note.

    This function cannot see validation targets. Empty/no-copy inputs retain
    the raw ranking. If every class would be excluded, retain the raw winner
    and mark the fallback; never manufacture an allowed class.
    """
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=object)
    if scores.ndim != 2 or scores.shape != (len(validation_notes), len(classes)):
        raise ValueError("scores must have one row per validation note and one column per class.")
    if not np.isfinite(scores).all():
        raise ValueError("Raw class scores must be finite.")
    raw_idx = scores.argmax(axis=1)
    output_idx = raw_idx.copy()
    diagnostics = []
    class_index = {str(label): i for i, label in enumerate(classes)}
    for row, note in enumerate(validation_notes):
        key = canonical_note(note)
        seen = seen_labels.get(key, set()) if key else set()
        excluded = sorted(label for label in seen if label in class_index)
        blocked = np.zeros(len(classes), dtype=bool)
        for label in excluded:
            blocked[class_index[label]] = True
        all_excluded = bool(blocked.all())
        if blocked.any() and not all_excluded:
            output_idx[row] = np.where(blocked, -np.inf, scores[row]).argmax()
        ranked = np.argsort(-scores[row], kind="stable")
        diagnostics.append({
            "nonempty_transcription": bool(key), "has_train_copy": bool(seen),
            "seen_training_classes": json.dumps(excluded, ensure_ascii=False),
            "n_excluded_classes": len(excluded), "raw_prediction_excluded": bool(blocked[raw_idx[row]]),
            "fallback_all_excluded": all_excluded,
            "changed": bool(output_idx[row] != raw_idx[row]),
            "raw_choice_score": float(scores[row, raw_idx[row]]),
            "rule_choice_score": float(scores[row, output_idx[row]]),
            "rule_choice_rank_before_exclusion": int(np.flatnonzero(ranked == output_idx[row])[0]) + 1,
        })
    return classes[raw_idx], classes[output_idx], pd.DataFrame(diagnostics)


def metric_rows(actual, predicted, classes):
    if not len(actual):
        return {"n": 0, "accuracy": None, "f1_macro": None, "min_f1": None}, None
    report = classification_report(actual, predicted, labels=classes, output_dict=True, zero_division=0)
    worst = min(classes, key=lambda label: report[label]["f1-score"])
    acc = float(accuracy_score(actual, predicted))
    minimum = float(report[worst]["f1-score"])
    return {
        "n": len(actual), "accuracy": acc,
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "min_f1": minimum, "worst_class": worst,
        "zero_f1_classes": sum(report[label]["f1-score"] == 0 for label in classes),
        "meets_thresholds": acc > .49 and minimum >= .25,
    }, report


def write_csv(frame, name):
    frame.to_csv(OUT / name, sep=";", encoding="utf-8-sig", index=False)


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def diagnostics_summary(frame):
    raw_correct = frame["raw_correct"]
    ruled_correct = frame["rule_correct"]
    copied = frame["has_train_copy"]
    return {
        "n": len(frame), "has_train_copy": int(copied.sum()),
        "copied_with_true_class_seen": int((copied & frame["true_class_seen_in_training_copies"]).sum()),
        "copied_with_true_class_unseen": int((copied & ~frame["true_class_seen_in_training_copies"]).sum()),
        "raw_winner_excluded": int(frame["raw_prediction_excluded"].sum()),
        "changed": int(frame["changed"].sum()), "fallback_all_excluded": int(frame["fallback_all_excluded"].sum()),
        "corrected": int((~raw_correct & ruled_correct).sum()),
        "harmed": int((raw_correct & ~ruled_correct).sum()),
        "changed_but_still_wrong": int((frame["changed"] & ~raw_correct & ~ruled_correct).sum()),
        "raw_correct": int(raw_correct.sum()), "rule_correct": int(ruled_correct.sum()),
        "net_corrected": int(ruled_correct.sum() - raw_correct.sum()),
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    source_names = ("train.csv", "augmentation.py", "models.py", "data_preparation.py", "audit_duplicate_exclusion.py")
    hashes = {name: sha256(ROOT / name) for name in source_names}
    records, _ = repair_training_csv(ROOT / "train.csv")
    train = pd.DataFrame(records).reset_index(drop=True)
    y = train[LABEL].astype(str)
    classes = sorted(y.unique())
    group_ids = identity_groups(train)
    plan = {
        "status": "train-only diagnostic audit; not independent evaluation or official-test accuracy",
        "model_ids": list(MODEL_IDS), "experiments": [list(e) for e in EXPERIMENTS],
        "n_folds": N_FOLDS, "shuffle": True, "original_records": len(train),
        "models": {"NB": "description TF-IDF + Porter stemming MultinomialNB alpha1 empirical priors",
                   "B": "two-field TF-IDF balanced logistic regression C=.25",
                   "SVC": "same two-field TF-IDF balanced LinearSVC C=.25"},
        "rule": "Exclude all classes observed for the exact canonical transcription in the training fold; next-largest score; nonempty notes only; all-excluded fallback raw",
        "canonicalization": "whitespace collapsed, strip, lowercase",
        "scores": {"NB": "predict_log_proba", "B": "decision_function", "SVC": "decision_function"},
        "class_weights": "training-original balanced weights, sum equals original training rows; no augmentation",
        "critical_assumption": "For a repeated note, a held-out row tends to have a class absent among that note's training copies; this is dataset construction, not a clinical law.",
        "leakage_controls": [
            "Split original rows before estimating vocabulary, IDF, class weights or note-label lookup.",
            "Rule helper accepts training labels and validation texts, never validation labels.",
            "Validation labels are used afterwards only to measure accuracy and assumption violations.",
            "No official test, selected final model, or production predictions are read.",
        ],
        "sources_sha256": hashes, "LR_C": LR_C,
        "versions": {"python": sys.version.split()[0], **{p: metadata.version(p) for p in (
            "pandas", "numpy", "scikit-learn", "nltk", "threadpoolctl",
        )}},
    }
    save_json("plan.json", plan)
    started = time.perf_counter()
    all_predictions, fold_scores, pooled_scores, subgroup_scores, class_scores, diagnosis_rows, fold_tables, warnings_log = [], [], [], [], [], [], [], []
    for protocol, seed in EXPERIMENTS:
        split = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed) if protocol == "stratified" else StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        folds = list(split.split(train, y, group_ids)) if protocol == "grouped" else list(split.split(train, y))
        experiment_predictions = []
        for fold, (tr, va) in enumerate(folds, 1):
            assert set(tr).isdisjoint(va)
            if protocol == "grouped":
                assert not (set(group_ids.iloc[tr]) & set(group_ids.iloc[va]))
            original_training = train.iloc[tr].reset_index(drop=True)
            validation = train.iloc[va].reset_index(drop=True)
            assert set(original_training[LABEL]) == set(classes)
            for record_id in validation["record_id"]:
                fold_tables.append({"protocol": protocol, "seed": seed, "record_id": record_id, "fold": fold})
            lookup = training_note_labels(original_training["transcription"], original_training[LABEL])
            batch = prepare_training(original_training, "B", seed=seed, fold=fold)
            feature_extractor = clone(build_condition_model().named_steps["features"])
            Z_train = feature_extractor.fit_transform(batch.X, batch.y)
            Z_val = feature_extractor.transform(clean_inputs(validation))
            for model_id in MODEL_IDS:
                fit_started = time.perf_counter()
                model = build_evaluated_model(model_id)
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                    warnings.simplefilter("always")
                    if model_id == "NB":
                        model.fit(prediction_input(original_training, "NB"), original_training[LABEL])
                        scores = model.predict_log_proba(prediction_input(validation, "NB"))
                        expected_raw = model.predict(prediction_input(validation, "NB"))
                        model_classes = model.classes_
                    else:
                        classifier = model.named_steps["classifier"]
                        classifier.fit(Z_train, batch.y, sample_weight=batch.sample_weight)
                        scores = classifier.decision_function(Z_val)
                        expected_raw = classifier.predict(Z_val)
                        model_classes = classifier.classes_
                    for item in caught:
                        warnings_log.append({"protocol": protocol, "seed": seed, "fold": fold, "model_id": model_id, "warning": str(item.message)})
                raw, ruled, detail = exclude_seen_classes(scores, model_classes, validation["transcription"], lookup)
                assert np.array_equal(raw, expected_raw)
                detail.insert(0, "protocol", protocol)
                detail.insert(1, "seed", seed)
                detail.insert(2, "fold", fold)
                detail.insert(3, "model_id", model_id)
                detail.insert(4, "record_id", validation["record_id"].to_numpy())
                detail["true_class"] = validation[LABEL].to_numpy()
                detail["raw_prediction"], detail["rule_prediction"] = raw, ruled
                detail["raw_correct"] = raw == validation[LABEL].to_numpy()
                detail["rule_correct"] = ruled == validation[LABEL].to_numpy()
                # Preserve scores for separately labelled sensitivity analyses
                # without refitting or changing the primary predeclared rule.
                for class_idx, label in enumerate(model_classes):
                    detail[f"score::{label}"] = scores[:, class_idx]
                detail["true_class_seen_in_training_copies"] = [
                    str(label) in lookup.get(canonical_note(note), set()) if canonical_note(note) else False
                    for note, label in zip(validation["transcription"], validation[LABEL])
                ]
                experiment_predictions.append(detail)
                for mode, pred in [("raw", raw), ("rule", ruled)]:
                    metrics, _ = metric_rows(validation[LABEL], pred, classes)
                    fold_scores.append({"protocol": protocol, "seed": seed, "fold": fold, "model_id": model_id,
                                        "mode": mode, "seconds_fit_both_modes": time.perf_counter() - fit_started, **metrics})
                print(f"{protocol} seed {seed} fold {fold}/{N_FOLDS} {model_id}: raw={detail.raw_correct.mean():.2%} rule={detail.rule_correct.mean():.2%} changed={detail.changed.sum()}", flush=True)
        current = pd.concat(experiment_predictions, ignore_index=True)
        for model_id in MODEL_IDS:
            frame = current.loc[current.model_id.eq(model_id)].copy()
            assert len(frame) == len(train) and frame.record_id.nunique() == len(train)
            for mode in ["raw", "rule"]:
                metrics, report = metric_rows(frame.true_class, frame[f"{mode}_prediction"], classes)
                pooled_scores.append({"protocol": protocol, "seed": seed, "model_id": model_id, "mode": mode, **metrics})
                for label in classes:
                    class_scores.append({"protocol": protocol, "seed": seed, "model_id": model_id, "mode": mode,
                                         "class": label, "precision": report[label]["precision"],
                                         "recall": report[label]["recall"], "f1": report[label]["f1-score"], "support": int(report[label]["support"])})
                masks = {
                    "with_train_copy": frame.has_train_copy,
                    "without_train_copy": ~frame.has_train_copy,
                    "copy_true_class_unseen": frame.has_train_copy & ~frame.true_class_seen_in_training_copies,
                    "copy_true_class_seen": frame.has_train_copy & frame.true_class_seen_in_training_copies,
                    "changed": frame.changed, "unchanged": ~frame.changed,
                    "fallback_all_excluded": frame.fallback_all_excluded,
                }
                for name, mask in masks.items():
                    sub_metrics, _ = metric_rows(frame.loc[mask, "true_class"], frame.loc[mask, f"{mode}_prediction"], classes)
                    subgroup_scores.append({"protocol": protocol, "seed": seed, "model_id": model_id, "mode": mode, "subgroup": name, **sub_metrics})
            diagnosis_rows.append({"protocol": protocol, "seed": seed, "model_id": model_id, **diagnostics_summary(frame)})
        all_predictions.append(current)
        write_csv(pd.DataFrame(pooled_scores), "summary.csv")
        write_csv(pd.DataFrame(fold_scores), "metrics_per_fold.csv")
        write_csv(pd.DataFrame(class_scores), "metrics_per_class.csv")
        write_csv(pd.DataFrame(subgroup_scores), "metrics_by_subgroup.csv")
        write_csv(pd.DataFrame(diagnosis_rows), "rule_diagnostics.csv")
        write_csv(pd.DataFrame(fold_tables), "fold_assignment.csv")
        joined = pd.concat(all_predictions, ignore_index=True)
        write_csv(joined, "oof_predictions.csv")
        write_csv(joined.loc[joined.changed], "changed_cases.csv")
        print("Completed", protocol, seed, flush=True)
        print(pd.DataFrame(pooled_scores).query("protocol == @protocol and seed == @seed").to_string(index=False), flush=True)
    observed_end_hashes = {name: sha256(ROOT / name) for name in source_names}
    save_json("experiment.json", {
        **plan, "state": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "warnings": warnings_log,
        "sources_unchanged_at_completion": {name: hashes[name] == observed_end_hashes[name] for name in source_names},
        "interpretation": "A gain under row splits measures exploitation of fold-local note/category complementarity. It is not a measurement of official-test accuracy or general clinical specialization.",
    })
    write_csv(pd.DataFrame(warnings_log, columns=["protocol", "seed", "fold", "model_id", "warning"]), "warnings.csv")


if __name__ == "__main__":
    run()
