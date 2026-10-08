r"""Compare input fields while keeping logistic regression and TF-IDF settings fixed.

Run from the project folder:
    .venv\Scripts\python.exe compare_logistic_fields.py

The notebook can pass its own folds and settings to compare_logistic_fields().
Only labelled training cases are used; no model or test predictions are replaced.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

from config import LABEL, MIN_ACCURACY, MIN_CLASS_F1, N_FOLDS, ROOT, SEED, TRAIN_FILE
from data import load_train
from data_preparation import sha256
from models import logistic_regression


TEXT_FIELDS = ("description", "sample_name", "transcription")
FIELD_SETS = tuple(
    fields for size in range(1, len(TEXT_FIELDS) + 1)
    for fields in combinations(TEXT_FIELDS, size)
)


def _metrics(actual, predicted, classes) -> tuple[dict, dict]:
    report = classification_report(
        actual, predicted, labels=classes, output_dict=True, zero_division=0
    )
    scores = {"accuracy": float(accuracy_score(actual, predicted))}
    for average in ("macro", "weighted"):
        for metric in ("precision", "recall", "f1"):
            key = "f1-score" if metric == "f1" else metric
            scores[f"{metric}_{average}"] = float(report[f"{average} avg"][key])
    scores["min_f1"] = min(float(report[label]["f1-score"]) for label in classes)
    return scores, report


def compare_logistic_fields(
    train: pd.DataFrame,
    *,
    field_sets: Sequence[Sequence[str]] = FIELD_SETS,
    splits=None,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    cv_mode: str = "stratified",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate field combinations with one prediction per case outside its training.

    Reuses the exact splits for every combination. Each fit learns its own
    vocabulary and IDF only from that fold's training data. Metrics are computed
    from all out-of-fold predictions together, with separate fold/class tables.
    """
    train = train.reset_index(drop=True)
    y = train[LABEL].astype(str)
    classes = sorted(y.unique())
    field_sets = tuple(tuple(fields) for fields in field_sets)
    if not field_sets or len(set(field_sets)) != len(field_sets):
        raise ValueError("Choose at least one field combination, without repetitions.")
    # Validate selectors through the model factory before running experiments.
    for fields in field_sets:
        logistic_regression(fields=fields)
    if splits is None:
        if cv_mode != "stratified":
            raise ValueError("Pass the notebook's grouped splits when cv_mode is grouped.")
        if y.value_counts().min() < n_folds:
            raise ValueError("The smallest class has fewer cases than n_folds.")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(train, y))
    else:
        splits = list(splits)
    if len(splits) < 2:
        raise ValueError("At least two folds are required.")

    visits = np.zeros(len(y), dtype=int)
    assignment = np.zeros(len(y), dtype=int)
    all_indices = set(range(len(y)))
    for fold, (train_idx, validation_idx) in enumerate(splits, start=1):
        train_set, validation_set = set(train_idx), set(validation_idx)
        if (
            len(train_set) != len(train_idx) or len(validation_set) != len(validation_idx)
            or train_set & validation_set or train_set | validation_set != all_indices
            or not train_set or not validation_set
        ):
            raise ValueError("Each fold must partition the data without overlap or repeated indices.")
        if set(y.iloc[train_idx]) != set(classes):
            raise ValueError("A class is missing from a training fold.")
        visits[validation_idx] += 1
        assignment[validation_idx] = fold
    if not np.all(visits == 1):
        raise ValueError("Every case must appear in validation exactly once.")

    used_fields = sorted({field for fields in field_sets for field in fields})
    clean = train[used_fields].fillna("").astype(str).apply(
        lambda column: column.str.replace(r"\s+", " ", regex=True).str.strip()
    )
    predictions = pd.DataFrame({
        "record_id": train["record_id"] if "record_id" in train else np.arange(1, len(train) + 1),
        "classe_real": y, "fold": assignment,
    })
    summary_rows, class_rows, fold_rows = [], [], []
    for fields in field_sets:
        name = " + ".join(fields)
        inputs = clean[list(fields)]
        predicted = np.full(len(y), "", dtype=object)
        started = time.perf_counter()
        for fold, (train_idx, validation_idx) in enumerate(splits, start=1):
            model = logistic_regression(
                fields=fields, ngram_range=ngram_range, min_df=min_df,
                sublinear_tf=sublinear_tf, C=C, class_weight=class_weight,
                max_iter=max_iter, random_state=seed,
            )
            model.fit(inputs.iloc[train_idx], y.iloc[train_idx])
            fold_predictions = model.predict(inputs.iloc[validation_idx])
            predicted[validation_idx] = fold_predictions
            scores, _ = _metrics(y.iloc[validation_idx], fold_predictions, classes)
            fold_rows.append({"fields": name, "fold": fold, **scores})
            print(
                f"LR | {name} | fold {fold}/{len(splits)} | "
                f"accuracy={scores['accuracy']:.1%} | F1 macro={scores['f1_macro']:.1%}",
                flush=True,
            )
        if np.any(predicted == ""):
            raise RuntimeError("Missing out-of-fold predictions.")
        scores, report = _metrics(y, predicted, classes)
        worst_class = min(classes, key=lambda label: report[label]["f1-score"])
        summary_rows.append({
            "fields": name, **scores, "worst_class": worst_class,
            "zero_f1_classes": sum(report[label]["f1-score"] == 0 for label in classes),
            "meets_criteria": scores["accuracy"] > MIN_ACCURACY and scores["min_f1"] >= MIN_CLASS_F1,
            "seconds": time.perf_counter() - started,
        })
        for label in classes:
            class_rows.append({
                "fields": name, "classe": label,
                "precision": report[label]["precision"], "recall": report[label]["recall"],
                "f1": report[label]["f1-score"], "support": int(report[label]["support"]),
            })
        predictions[name] = predicted

    result = {
        "comparison": pd.DataFrame(summary_rows).set_index("fields"),
        "per_class": pd.DataFrame(class_rows),
        "per_fold": pd.DataFrame(fold_rows),
        "predictions": predictions,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_names = {
            "comparison": "comparacao_campos.csv", "per_class": "metricas_por_classe.csv",
            "per_fold": "metricas_por_fold.csv", "predictions": "previsoes_validacao.csv",
        }
        for key, frame in result.items():
            frame.to_csv(
                output_dir / file_names[key], sep=";", encoding="utf-8-sig", index=key == "comparison"
            )
        import importlib.metadata as metadata
        import sys

        experiment = {
            "model": "logistic_regression", "field_sets": [list(fields) for fields in field_sets],
            "cv_mode": cv_mode, "n_folds": len(splits), "seed": seed,
            "ngram_range": list(ngram_range), "min_df": min_df, "sublinear_tf": sublinear_tf,
            "C": C, "class_weight": class_weight, "max_iter": max_iter,
            "train_records": len(y), "classes": classes,
            "source_sha256": sha256(TRAIN_FILE),
            "versions": {"python": sys.version.split()[0], **{
                package: metadata.version(package) for package in ("pandas", "numpy", "scikit-learn")
            }},
            "metrics_source": "pooled out-of-fold predictions",
        }
        (output_dir / "experiencia.json").write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    output_dir = ROOT / "outputs" / "comparacao_campos_lr"
    result = compare_logistic_fields(load_train(), output_dir=output_dir)
    print(result["comparison"][["accuracy", "f1_macro", "min_f1", "meets_criteria"]].to_string(
        float_format=lambda value: f"{value:.1%}"
    ))
    print(f"Saved metrics and out-of-fold predictions to {output_dir}")


if __name__ == "__main__":
    main()
