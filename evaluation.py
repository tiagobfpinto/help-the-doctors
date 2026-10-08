"""Cross-validation and the two pass criteria of the automatic evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import MIN_ACCURACY, MIN_CLASS_F1, N_FOLDS, SEED


def out_of_fold_predictions(model: BaseEstimator, texts: pd.Series, labels: pd.Series) -> np.ndarray:
    """Predict every training case with a copy of the model that never saw it.

    The data is split into N_FOLDS parts; each part is predicted by a model
    trained on the others. Folds are stratified (each has the same class mix)
    and shuffled, because the CSV is grouped by specialty.
    """
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(model, texts, labels, cv=folds)


def summary(labels: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Accuracy, mean per-class F1 and lowest per-class F1, for comparison tables."""
    per_class_f1 = f1_score(labels, predicted, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "mean_f1": per_class_f1.mean(),
        "min_f1": per_class_f1.min(),
    }


def report(labels: pd.Series, predicted: np.ndarray) -> bool:
    """Print per-class scores and check both criteria. Returns True if both pass."""
    print(classification_report(labels, predicted, digits=3, zero_division=0))

    classes = sorted(labels.unique())
    per_class_f1 = f1_score(labels, predicted, labels=classes, average=None, zero_division=0)
    worst = per_class_f1.argmin()
    accuracy = accuracy_score(labels, predicted)

    print(f"Accuracy:  {accuracy:.1%}  (needs > {MIN_ACCURACY:.0%})")
    print(f"Lowest F1: {per_class_f1[worst]:.1%} for {classes[worst]}  (needs >= {MIN_CLASS_F1:.0%})")
    return accuracy > MIN_ACCURACY and per_class_f1.min() >= MIN_CLASS_F1
