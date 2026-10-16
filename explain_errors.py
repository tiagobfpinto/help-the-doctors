"""Reproduce three selected development errors and decompose their SVC margins.

Run from the project root. Only labelled training rows and stored OOF fold
assignments are read. No saved final model or unlabelled test is used.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from augmentation import fit_evaluated_model, prediction_input
from data_preparation import repair_training_csv, sha256

OUT = ROOT / "outputs" / "augmentation_controlada"
RECORD_IDS = (722, 308, 298)


def add_training_word_counts(train, oof, checks):
    """Audit field-specific support without fitting or reading validation text."""
    for check in checks:
        training_input = prediction_input(train.loc[oof["fold"].ne(check["fold"])], "SVC")
        check["training_document_counts_by_field"] = {
            term: {field: int(training_input[field].str.contains(
                rf"\b{term}\b", case=False, regex=True
            ).sum()) for field in training_input}
            for term in ["acne", "vulgaris", "orif", "radius", "rash", "skin"]
        }


def main():
    records, _ = repair_training_csv(ROOT / "train.csv")
    train = pd.DataFrame(records).reset_index(drop=True)
    oof = pd.read_csv(OUT / "oof_predictions.csv", sep=";")
    assert train.record_id.tolist() == oof.record_id.tolist()
    saved = json.loads((OUT / "experiment.json").read_text(encoding="utf-8"))
    for name in ("train.csv", "augmentation.py", "models.py", "data_preparation.py"):
        assert sha256(ROOT / name) == saved["sources_sha256"][name]
    selected_folds = sorted(oof.loc[oof.record_id.isin(RECORD_IDS), "fold"].unique())
    rows, checks = [], []
    for fold in selected_folds:
        is_validation = oof["fold"].eq(fold)
        training = train.loc[~is_validation].reset_index(drop=True)
        validation = train.loc[is_validation]
        model, _ = fit_evaluated_model(training, "SVC", seed=0, fold=int(fold))
        predicted = model.predict(prediction_input(validation, "SVC"))
        assert np.array_equal(predicted, oof.loc[is_validation, "SVC"].to_numpy())
        features = model.named_steps["features"]
        classifier = model.named_steps["classifier"]
        names = features.get_feature_names_out()
        for record_id in RECORD_IDS:
            chosen = train["record_id"].eq(record_id) & is_validation
            if not chosen.any():
                continue
            case = train.loc[chosen]
            z = features.transform(prediction_input(case, "SVC")).tocoo()
            decisions = model.decision_function(prediction_input(case, "SVC"))[0]
            true_label = case["medical_specialty"].iloc[0]
            predicted_label = oof.loc[chosen, "SVC"].iloc[0]
            true_idx = np.flatnonzero(classifier.classes_ == true_label)[0]
            pred_idx = np.flatnonzero(classifier.classes_ == predicted_label)[0]
            delta_coef = classifier.coef_[pred_idx] - classifier.coef_[true_idx]
            contributions = z.data * delta_coef[z.col]
            intercept_delta = float(classifier.intercept_[pred_idx] - classifier.intercept_[true_idx])
            actual_margin = float(decisions[pred_idx] - decisions[true_idx])
            reconstructed = float(contributions.sum() + intercept_delta)
            assert np.isclose(actual_margin, reconstructed, rtol=1e-10, atol=1e-10)
            case_rows = []
            for col, value, contribution in zip(z.col, z.data, contributions):
                field, feature = names[col].split("__", 1)
                case_rows.append({
                    "record_id": record_id, "fold": int(fold), "true_class": true_label,
                    "predicted_class": predicted_label, "component_type": "feature",
                    "field": field, "feature": feature, "tfidf": float(value),
                    "coefficient_predicted": float(classifier.coef_[pred_idx, col]),
                    "coefficient_true": float(classifier.coef_[true_idx, col]),
                    "coefficient_difference": float(delta_coef[col]),
                    "contribution_to_predicted_minus_true": float(contribution),
                })
            case_rows.sort(key=lambda row: abs(row["contribution_to_predicted_minus_true"]), reverse=True)
            pos_rank = neg_rank = 0
            for rank, row in enumerate(case_rows, 1):
                row["rank_absolute"] = rank
                if row["contribution_to_predicted_minus_true"] > 0:
                    pos_rank += 1
                    row["rank_favors_predicted"] = pos_rank
                    row["rank_favors_true"] = None
                else:
                    neg_rank += 1
                    row["rank_favors_predicted"] = None
                    row["rank_favors_true"] = neg_rank
                rows.append(row)
            rows.append({
                "record_id": record_id, "fold": int(fold), "true_class": true_label,
                "predicted_class": predicted_label, "component_type": "intercept",
                "field": "intercept", "feature": "intercept", "tfidf": 1.0,
                "coefficient_predicted": float(classifier.intercept_[pred_idx]),
                "coefficient_true": float(classifier.intercept_[true_idx]),
                "coefficient_difference": intercept_delta,
                "contribution_to_predicted_minus_true": intercept_delta,
            })
            check = {
                "record_id": record_id, "fold": int(fold), "true_class": true_label,
                "predicted_class": predicted_label, "sample_name": case["sample_name"].iloc[0],
                "validation_fold_predictions_exactly_reproduced": True,
                "predicted_decision_score": float(decisions[pred_idx]),
                "true_decision_score": float(decisions[true_idx]),
                "predicted_minus_true_margin": actual_margin,
                "sum_feature_contributions": float(contributions.sum()),
                "intercept_contribution": intercept_delta, "reconstructed_margin": reconstructed,
                "absolute_reconstruction_error": abs(actual_margin - reconstructed),
                "clinical_title_terms_in_vocabulary": {
                    term: [str(name) for name in names if name.split("__", 1)[1] == term]
                    for term in ["acne", "vulgaris", "orif", "radius", "rash", "skin"]
                },
            }
            checks.append(check)
            print(json.dumps({key: value for key, value in check.items() if key != "clinical_title_terms_in_vocabulary"}, ensure_ascii=False))
            print("Top features favoring predicted:", [(r["field"], r["feature"], r["contribution_to_predicted_minus_true"]) for r in case_rows if r.get("rank_favors_predicted") is not None][:8])
            print("Top features favoring assigned label:", [(r["field"], r["feature"], r["contribution_to_predicted_minus_true"]) for r in case_rows if r.get("rank_favors_true") is not None][:8])
    add_training_word_counts(train, oof, checks)
    pd.DataFrame(rows).to_csv(OUT / "error_feature_attributions.csv", sep=";", index=False, encoding="utf-8-sig")
    info = {
        "model_id": "SVC", "record_ids": list(RECORD_IDS), "folds_refitted": [int(v) for v in selected_folds],
        "source_sha256": saved["source_sha256"], "attribution_script_sha256": sha256(Path(__file__)),
        "method": "x_j * (coef_predicted_j - coef_assigned_j); intercept difference separate; OvR decision scores",
        "checks": checks,
        "interpretation": "Exact local linear-score decomposition, not causal evidence or a medical explanation; feature associations may reflect limited training data and label overlap.",
    }
    (OUT / "error_feature_attributions_metadata.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
