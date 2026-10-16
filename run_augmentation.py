r"""Run the frozen seven-configuration, five-fold development comparison.

    .venv\Scripts\python.exe run_augmentation.py

This script reads labelled train.csv only. It never reads the unlabelled test,
never writes results.txt and never saves a fitted final model. Selection on
these development OOF results is not an independent test evaluation.
"""

from __future__ import annotations

import argparse
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from augmentation import (
    ARTICLE_RATE, AUGMENTED_CLASSES, CONDITION_NAMES, CONDITIONS, LABEL, LR_C,
    MAX_ITER, TEXT_FIELDS, build_condition_model, build_evaluated_model,
    clean_inputs, identity_groups, prediction_input, prepare_training,
)
from data_preparation import repair_training_csv, sha256

ROOT = Path(__file__).resolve().parent
MODEL_IDS = (*CONDITIONS, "NB", "SVC")
SEED = 0
N_FOLDS = 5


def scores(actual, predicted, classes):
    result = classification_report(actual, predicted, labels=classes, output_dict=True, zero_division=0)
    worst = min(classes, key=lambda label: result[label]["f1-score"])
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": result["macro avg"]["precision"],
        "recall_macro": result["macro avg"]["recall"],
        "f1_macro": result["macro avg"]["f1-score"],
        "f1_weighted": result["weighted avg"]["f1-score"],
        "min_f1": result[worst]["f1-score"], "worst_class": worst,
        "zero_f1_classes": sum(result[label]["f1-score"] == 0 for label in classes),
        "meets_criteria": accuracy_score(actual, predicted) > .49 and result[worst]["f1-score"] >= .25,
    }
    return metrics, result


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(frame, path, *, index=False):
    frame.to_csv(path, sep=";", encoding="utf-8-sig", index=index)


def design_spec(train):
    return {
        "status": "fixed exploratory development comparison; not an independent final test",
        "model_ids": list(MODEL_IDS), "conditions": CONDITION_NAMES,
        "source_sha256": sha256(ROOT / "train.csv"),
        "sources_sha256": {name: sha256(ROOT / name) for name in (
            "train.csv", "augmentation.py", "run_augmentation.py", "models.py", "data_preparation.py",
        )},
        "records": len(train), "class_counts": train[LABEL].value_counts().to_dict(),
        "fields": list(TEXT_FIELDS), "seed": SEED,
        "cv": {"type": "StratifiedGroupKFold", "n_splits": N_FOLDS, "shuffle": True, "random_state": SEED},
        "group_rule": "lowercase whitespace-normalized transcription; if empty, normalized sample_name; all-inputs-empty rows unique",
        "tfidf": {"per_field": True, "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True,
                  "fit_scope": "only original + added rows of each arm's training fold"},
        "logistic_regression": {"C": LR_C, "max_iter": MAX_ITER, "solver": "lbfgs", "penalty": "L2"},
        "linear_svc": {"C": LR_C, "max_iter": MAX_ITER, "loss": "squared_hinge", "penalty": "L2", "multiclass": "ovr"},
        "naive_bayes": {"fields": ["description"], "Porter_stemming": True, "alpha": 1.0, "fit_prior": True},
        "augmentation": {
            "name": "conservative article dropout, EDA-inspired random deletion",
            "field": "transcription", "sample_name": "preserved",
            "eligible_classes": list(AUGMENTED_CLASSES), "eligibility": "fixed original classes with n < 100",
            "variants_per_accepted_parent": 1, "article_regex": r"(?u)\bthe\b with IGNORECASE",
            "k": "max(1, floor(rate * number_of_article_occurrences))", "rate": ARTICLE_RATE,
            "random_seed": "int.from_bytes(SHA256(f'{seed}|{fold}|{record_id}|transcription').digest(), 'big')",
            "rejections": ["no_article", "empty_after_deletion"],
            "copies_control": "E duplicates exactly the parents accepted by C/D, once each, in identical order",
            "no_global_label_conflict_filter": True,
        },
        "weights": {
            "balanced_formula": "N_original / (K * original_training_count_class)",
            "original_only_estimation": True, "fixed_for_D_E": True,
            "normalization": "sum(sample_weight) = N_original in every arm; not mean 1 over augmented rows",
            "interpretation": "D/E match source reweighting; feature counts, IDF and min_df are refitted after additions",
        },
        "selection_rule": "among min_f1 >= .25: max accuracy, then macro F1, then min F1, then model_id; otherwise max macro F1, then accuracy, then min F1",
        "limitations": [
            "Previous pilots and field selection used these labelled data; this is development, not independent evaluation.",
            "Identity groups are not patient IDs and do not capture merely similar notes.",
            "Entirely missing model inputs receive unique groups and remain in every evaluation.",
            "One conservative augmentation and one setting per family do not rank augmentation strategies or optimized model families.",
            "Positive or negative augmentation results cannot be generalized to clinical paraphrasing.",
        ],
    }


def check_split(train, groups, train_idx, validation_idx):
    if set(groups.iloc[train_idx]) & set(groups.iloc[validation_idx]):
        raise AssertionError("A group crosses a training/validation boundary.")
    normalized = clean_inputs(train).apply(lambda col: col.str.lower())
    keys = normalized.apply(lambda row: tuple(row), axis=1)
    nonempty = normalized.ne("").any(axis=1)
    left = set(keys.iloc[train_idx][nonempty.iloc[train_idx]])
    right = set(keys.iloc[validation_idx][nonempty.iloc[validation_idx]])
    if left & right:
        raise AssertionError("Identical nonempty model inputs cross a boundary.")


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    records, _ = repair_training_csv(ROOT / "train.csv")
    train = pd.DataFrame(records).reset_index(drop=True)
    counts = train[LABEL].value_counts()
    if set(counts[counts < 100].index) != set(AUGMENTED_CLASSES):
        raise ValueError("Class counts changed: the frozen augmentation eligibility no longer matches this dataset.")
    y = train[LABEL].astype(str)
    classes = sorted(y.unique())
    design = design_spec(train)
    # Written before any model fit or outcome. Never change design after scores.
    save_json(output_dir / "design.json", design)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    save_json(output_dir / "run_state.json", {"state": "running", "started_utc": started_at})
    groups = identity_groups(train)
    folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train, y, groups))
    assignment = np.zeros(len(y), dtype=int)
    for fold, (tr, va) in enumerate(folds, 1):
        check_split(train, groups, tr, va)
        assert set(y.iloc[tr]) == set(classes)
        assignment[va] = fold
    visits = np.zeros(len(y), dtype=int)
    for _, va in folds:
        visits[va] += 1
    assert np.all(visits == 1)
    fold_frame = train[["record_id", LABEL]].copy()
    fold_frame["fold"], fold_frame["group_id"] = assignment, groups
    write_csv(fold_frame, output_dir / "fold_assignment.csv")
    predictions = {name: np.full(len(y), "", dtype=object) for name in MODEL_IDS}
    fold_rows, audit_frames, weight_rows, examples, warning_rows = [], [], [], [], []
    print(f"Frozen design saved. {len(train)} original cases, {len(classes)} classes, {len(folds)} grouped folds.", flush=True)

    for fold, (tr, va) in enumerate(folds, 1):
        train_fold, validation = train.iloc[tr].reset_index(drop=True), train.iloc[va]
        batches = {name: prepare_training(train_fold, name, seed=SEED, fold=fold) for name in CONDITIONS}
        np.testing.assert_allclose(batches["D"].sample_weight, batches["E"].sample_weight)
        pd.testing.assert_series_equal(batches["D"].y, batches["E"].y)
        audit = batches["D"].audit.copy()
        accepted_ids = set(batches["D"].children["parent_record_id"])
        assert accepted_ids <= set(train.iloc[tr]["record_id"])
        assert accepted_ids.isdisjoint(set(validation["record_id"]))
        audit_frames.append(audit)
        children = batches["D"].children
        for _, child in children.groupby(LABEL, sort=True).head(2).iterrows():
            parent = train_fold.set_index("record_id").loc[child["parent_record_id"]]
            examples.append({
                "fold": fold, "parent_record_id": child["parent_record_id"], "class": child[LABEL],
                "sample_name": parent["sample_name"], "original_transcription": parent["transcription"],
                "augmented_transcription": child["transcription"],
            })
        # Shared fitted features for arms with exactly identical training texts.
        feature_cache = {}
        for key, name in [("original", "A"), ("dropout", "C"), ("copies", "E")]:
            features = clone(build_condition_model().named_steps["features"])
            feature_cache[key] = (
                features.fit_transform(batches[name].X, batches[name].y),
                features.transform(clean_inputs(validation)),
            )
        for model_id in MODEL_IDS:
            fit_started = time.perf_counter()
            model = build_evaluated_model(model_id)
            if model_id in CONDITIONS:
                batch = batches[model_id]
            else:
                batch = batches["A"] if model_id == "NB" else batches["B"]
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always", ConvergenceWarning)
                if model_id == "NB":
                    model.fit(prediction_input(train_fold, "NB"), batch.y,
                              classifier__sample_weight=batch.sample_weight)
                    pred = model.predict(prediction_input(validation, "NB"))
                    n_features = len(model.named_steps["counts"].vocabulary_)
                else:
                    key = "dropout" if model_id in {"C", "D"} else "copies" if model_id == "E" else "original"
                    Z_train, Z_validation = feature_cache[key]
                    classifier = model.named_steps["classifier"]
                    classifier.fit(Z_train, batch.y, sample_weight=batch.sample_weight)
                    pred = classifier.predict(Z_validation)
                    n_features = Z_train.shape[1]
                for warning in caught:
                    warning_rows.append({"fold": fold, "model_id": model_id, "warning": str(warning.message)})
            predictions[model_id][va] = pred
            fold_scores, _ = scores(y.iloc[va], pred, classes)
            fold_rows.append({
                "model_id": model_id, "condition": model_id, "fold": fold,
                "training_originals": len(tr), "validation_originals": len(va),
                "fit_rows": len(batch.y), "features": n_features,
                "sample_weight_sum": float(batch.sample_weight.sum()),
                "seconds_fit_predict": time.perf_counter() - fit_started, **fold_scores,
            })
            weight_rows.append({"fold": fold, "model_id": model_id, **batch.metadata})
            print(f"Fold {fold}/{N_FOLDS} | {model_id} | accuracy={fold_scores['accuracy']:.3%} | macro F1={fold_scores['f1_macro']:.3%} | min F1={fold_scores['min_f1']:.3%}", flush=True)
            save_json(output_dir / "run_state.json", {
                "state": "running", "started_utc": started_at, "last_completed_fold": fold,
                "last_completed_model": model_id,
            })
        write_csv(pd.DataFrame(fold_rows), output_dir / "metrics_per_fold.csv")

    summary_rows, class_rows = [], []
    oof = fold_frame.copy()
    normalized = clean_inputs(train).apply(lambda col: col.str.lower())
    input_keys = normalized.apply(lambda row: json.dumps(tuple(row)), axis=1)
    input_label_counts = pd.DataFrame({"key": input_keys, "label": y}).groupby("key")["label"].nunique()
    for model_id, pred in predictions.items():
        assert np.all(pred != "")
        all_scores, report = scores(y, pred, classes)
        family = "MultinomialNB" if model_id == "NB" else "LinearSVC" if model_id == "SVC" else "LogisticRegression"
        summary_rows.append({"model_id": model_id, "condition": model_id, "model": family,
                             "protocol": "grouped_development", **all_scores})
        for label in classes:
            class_rows.append({"model_id": model_id, "condition": model_id, "class": label,
                               "precision": report[label]["precision"], "recall": report[label]["recall"],
                               "f1": report[label]["f1-score"], "support": int(report[label]["support"])})
        oof[model_id] = pred
        cm = confusion_matrix(y, pred, labels=classes)
        matrix = pd.DataFrame(cm, index=classes, columns=classes).rename_axis("true_class")
        write_csv(matrix, output_dir / f"confusion_{model_id}.csv", index=True)
        write_csv(matrix.div(matrix.sum(axis=1), axis=0), output_dir / f"confusion_normalized_{model_id}.csv", index=True)
        errors = train.loc[y.to_numpy() != pred, ["record_id", "description", "sample_name", "transcription"]].copy()
        errors["true_class"] = y.loc[errors.index]
        errors["predicted_class"] = pd.Series(pred, index=train.index).loc[errors.index]
        errors["fold"] = assignment[errors.index]
        errors["group_id"] = groups.loc[errors.index]
        errors["identical_input_multiple_labels"] = input_keys.loc[errors.index].map(input_label_counts).gt(1)
        write_csv(errors, output_dir / f"errors_{model_id}.csv")
        pair_counts = errors.groupby(["true_class", "predicted_class"]).size().sort_values(ascending=False)
        selected_errors = []
        for (actual, predicted_class), n in pair_counts.head(3).items():
            example = errors.loc[errors["true_class"].eq(actual) & errors["predicted_class"].eq(predicted_class)].iloc[0].to_dict()
            selected_errors.append({**example, "confusion_pair_count": int(n)})
        write_csv(pd.DataFrame(selected_errors), output_dir / f"error_examples_{model_id}.csv")
    audit = pd.concat(audit_frames, ignore_index=True)
    coverage = audit.groupby(["fold", "class"], sort=True).agg(
        eligible=("parent_record_id", "size"), changed=("changed", "sum"),
        removed_words=("removed_words", "sum"), original_words=("original_word_count", "sum"),
    ).reset_index()
    coverage["coverage"] = coverage["changed"] / coverage["eligible"]
    coverage["removed_fraction_of_eligible_words"] = coverage["removed_words"] / coverage["original_words"].clip(lower=1)
    write_csv(pd.DataFrame(summary_rows), output_dir / "summary.csv")
    write_csv(pd.DataFrame(class_rows), output_dir / "metrics_per_class.csv")
    write_csv(oof, output_dir / "oof_predictions.csv")
    write_csv(audit, output_dir / "augmentation_audit.csv")
    write_csv(coverage, output_dir / "augmentation_coverage.csv")
    write_csv(pd.DataFrame(examples), output_dir / "augmentation_examples.csv")
    write_csv(pd.DataFrame(warning_rows, columns=["fold", "model_id", "warning"]), output_dir / "warnings.csv")
    save_json(output_dir / "weights_per_fold.json", weight_rows)
    assert sha256(ROOT / "train.csv") == design["source_sha256"]
    experiment = {
        **design, "state": "complete", "started_utc": started_at,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "versions": {"python": sys.version.split()[0], **{p: metadata.version(p) for p in (
            "numpy", "pandas", "scikit-learn", "nltk", "threadpoolctl",
        )}},
        "augmentation_totals": {
            "eligible_training_parent_visits": len(audit), "accepted_training_parent_visits": int(audit["changed"].sum()),
            "rejections": audit.loc[~audit["changed"], "rejection"].value_counts().to_dict(),
            "removed_words": int(audit["removed_words"].sum()),
            "eligible_original_words": int(audit["original_word_count"].sum()),
            "removed_fraction_of_eligible_words": float(audit["removed_words"].sum() / max(1, audit["original_word_count"].sum())),
        },
        "warnings": warning_rows, "metrics_source": "pooled out-of-fold predictions of all original cases",
        "identical_nonempty_input_overlap_per_fold": [0] * N_FOLDS,
        "empty_model_inputs": int(normalized.eq("").all(axis=1).sum()),
    }
    save_json(output_dir / "experiment.json", experiment)
    save_json(output_dir / "run_state.json", {"state": "complete", "completed_utc": experiment["completed_utc"]})
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(f"Saved development comparison to {output_dir}; no unlabelled test was read.", flush=True)
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "augmentation_controlada")
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
