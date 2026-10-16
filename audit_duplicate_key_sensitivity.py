"""Post-hoc key sensitivity using saved OOF scores, with zero additional fits.

Run after audit_duplicate_exclusion.py completes. This diagnostic replaces the
primary nonempty-transcription key by the entire normalized model input pair.
It avoids treating short shared boilerplate with different titles as the same
case; title-only inputs are allowed, while wholly empty inputs remain untouched.
This is a separately labelled sensitivity, not an independent confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from augmentation import clean_inputs
from audit_duplicate_exclusion import (
    MODEL_IDS, diagnostics_summary, exclude_seen_classes, metric_rows, training_note_labels,
)
from data_preparation import repair_training_csv, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "auditoria_exclusao"


def pair_keys(frame):
    inputs = clean_inputs(frame).apply(lambda column: column.str.lower())
    return inputs.apply(lambda row: json.dumps(tuple(row), ensure_ascii=False) if row.ne("").any() else "", axis=1)


def main():
    experiment_path = OUT / "experiment.json"
    if not experiment_path.exists():
        raise RuntimeError("Complete the primary audit before running this sensitivity.")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    assert experiment["state"] == "complete"
    assert sha256(ROOT / "train.csv") == experiment["sources_sha256"]["train.csv"]
    source = pd.DataFrame(repair_training_csv(ROOT / "train.csv")[0]).reset_index(drop=True)
    classes = sorted(source.medical_specialty.unique())
    pair = pair_keys(source)
    primary = pd.read_csv(OUT / "oof_predictions.csv", sep=";")
    plan = {
        "status": "post-hoc sensitivity; no new model fits or independent test evaluation",
        "motivation": "The primary transcript-only lookup can match shared boilerplate under different sample names.",
        "key": "entire whitespace-normalized lowercase (sample_name, transcription); allow title-only; skip only if both fields empty",
        "rule": "same exclusion ranking and all-excluded fallback as primary audit",
        "primary_predictions_sha256": sha256(OUT / "oof_predictions.csv"),
        "source_sha256": sha256(ROOT / "train.csv"),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "pair_key_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    predictions, summaries, folds, per_class, diagnoses, subgroups = [], [], [], [], [], []
    for (protocol, seed), experiment_rows in primary.groupby(["protocol", "seed"], sort=False):
        for model_id in MODEL_IDS:
            model_rows = experiment_rows.loc[experiment_rows.model_id.eq(model_id)].copy()
            parts = []
            for fold, validation in model_rows.groupby("fold", sort=True):
                validation = validation.copy().reset_index(drop=True)
                held_out_ids = set(validation.record_id)
                training_mask = ~source.record_id.isin(held_out_ids)
                assert training_mask.sum() + len(validation) == len(source)
                seen = training_note_labels(pair.loc[training_mask], source.loc[training_mask, "medical_specialty"])
                val_source = source.set_index("record_id").loc[validation.record_id].reset_index()
                val_keys = pair_keys(val_source)
                scores = validation[[f"score::{label}" for label in classes]].to_numpy()
                raw, ruled, detail = exclude_seen_classes(scores, classes, val_keys, seen)
                assert np.array_equal(raw, validation.raw_prediction)
                for key in ["protocol", "seed", "fold", "model_id", "record_id", "true_class"]:
                    detail[key] = validation[key].to_numpy()
                detail["raw_prediction"], detail["rule_prediction"] = raw, ruled
                detail["raw_correct"] = raw == validation.true_class.to_numpy()
                detail["rule_correct"] = ruled == validation.true_class.to_numpy()
                detail["true_class_seen_in_training_copies"] = [
                    label in seen.get(key, set()) if key else False
                    for key, label in zip(val_keys, validation.true_class)
                ]
                detail["nonempty_transcription"] = val_source.transcription.fillna("").str.strip().ne("").to_numpy()
                detail["title_only_train_copy"] = detail.has_train_copy & ~detail.nonempty_transcription
                detail["primary_rule_prediction"] = validation.rule_prediction.to_numpy()
                detail["different_from_primary_rule"] = detail.rule_prediction.ne(detail.primary_rule_prediction)
                parts.append(detail)
                metrics, _ = metric_rows(detail.true_class, detail.rule_prediction, classes)
                folds.append({"protocol": protocol, "seed": int(seed), "fold": int(fold), "model_id": model_id, "mode": "pair_rule", **metrics})
            frame = pd.concat(parts, ignore_index=True)
            predictions.append(frame)
            metrics, report = metric_rows(frame.true_class, frame.rule_prediction, classes)
            summaries.append({"protocol": protocol, "seed": int(seed), "model_id": model_id, "mode": "pair_rule", **metrics})
            diagnoses.append({
                "protocol": protocol, "seed": int(seed), "model_id": model_id,
                "title_only_copies": int(frame.title_only_train_copy.sum()),
                "different_from_primary_rule": int(frame.different_from_primary_rule.sum()),
                **diagnostics_summary(frame),
            })
            for label in classes:
                per_class.append({"protocol": protocol, "seed": int(seed), "model_id": model_id, "mode": "pair_rule",
                                  "class": label, "precision": report[label]["precision"], "recall": report[label]["recall"],
                                  "f1": report[label]["f1-score"], "support": int(report[label]["support"])})
            for name, mask in {"with_pair_copy": frame.has_train_copy, "without_pair_copy": ~frame.has_train_copy,
                               "title_only_copy": frame.title_only_train_copy}.items():
                for mode in ["raw", "rule"]:
                    metrics, _ = metric_rows(frame.loc[mask, "true_class"], frame.loc[mask, f"{mode}_prediction"], classes)
                    subgroups.append({"protocol": protocol, "seed": int(seed), "model_id": model_id,
                                      "mode": mode, "subgroup": name, **metrics})
    for name, rows in [
        ("summary", summaries), ("metrics_per_fold", folds), ("metrics_per_class", per_class),
        ("diagnostics", diagnoses), ("metrics_by_subgroup", subgroups),
    ]:
        pd.DataFrame(rows).to_csv(OUT / f"pair_key_{name}.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "pair_key_oof_predictions.csv", sep=";", encoding="utf-8-sig", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(diagnoses).to_string(index=False))


if __name__ == "__main__":
    main()
