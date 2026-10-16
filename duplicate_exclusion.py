"""Fold-local/full-training exact-input decoder for the benchmark's annotation pattern.

Only the supplied original training rows define the lookup. Excluding a class
seen for the same input assumes that another copy receives another category;
this is a dataset-construction regularity, not a clinical law. The decoder is
inactive for unmatched notes and is not evidence of better unseen-note semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from augmentation import LABEL, clean_inputs, prediction_input


def exact_input_keys(frame: pd.DataFrame) -> pd.Series:
    """Canonical full model input; do not merge wholly unavailable inputs."""
    inputs = clean_inputs(frame).apply(lambda column: column.str.lower())
    return inputs.apply(
        lambda row: json.dumps(tuple(row), ensure_ascii=False) if row.ne("").any() else "", axis=1,
    )


@dataclass
class DecodingResult:
    predictions: np.ndarray
    raw_predictions: np.ndarray
    diagnostics: pd.DataFrame


@dataclass
class ExactInputExclusionIndex:
    label_sets: dict[str, tuple[str, ...]]
    original_training_rows: int

    def predict(self, scores, classes, frame: pd.DataFrame) -> DecodingResult:
        """Exclude all labels observed for an identical original training input.

        No target column is read from frame. If no input matches, or all classes
        would be excluded, preserve the raw winner. Stable class order resolves
        exact score ties consistently with numpy argmax and sklearn classifiers.
        """
        scores = np.asarray(scores, dtype=float)
        classes = np.asarray(classes, dtype=object)
        if scores.ndim != 2 or scores.shape != (len(frame), len(classes)):
            raise ValueError("scores must have one row per input and one column per class.")
        if not len(classes) or len(set(classes)) != len(classes) or not np.isfinite(scores).all():
            raise ValueError("Require distinct nonempty classes and finite raw scores.")
        keys = exact_input_keys(frame).tolist()
        positions = {str(label): i for i, label in enumerate(classes)}
        raw_idx = scores.argmax(axis=1)
        decoded_idx = raw_idx.copy()
        details = []
        for i, key in enumerate(keys):
            observed = self.label_sets.get(key, ()) if key else ()
            excluded = tuple(label for label in observed if label in positions)
            blocked = np.zeros(len(classes), dtype=bool)
            for label in excluded:
                blocked[positions[label]] = True
            all_blocked = bool(blocked.all())
            if blocked.any() and not all_blocked:
                decoded_idx[i] = np.where(blocked, -np.inf, scores[i]).argmax()
            ranked = np.argsort(-scores[i], kind="stable")
            details.append({
                "input_available": bool(key), "has_train_match": bool(observed),
                "seen_classes": json.dumps(excluded, ensure_ascii=False),
                "n_excluded_classes": len(excluded),
                "raw_prediction_excluded": bool(blocked[raw_idx[i]]),
                "fallback_all_excluded": all_blocked,
                "changed": bool(decoded_idx[i] != raw_idx[i]),
                "raw_choice_score": float(scores[i, raw_idx[i]]),
                "decoded_choice_score": float(scores[i, decoded_idx[i]]),
                "decoded_choice_raw_rank": int(np.flatnonzero(ranked == decoded_idx[i])[0]) + 1,
            })
        return DecodingResult(classes[decoded_idx], classes[raw_idx], pd.DataFrame(details))


def fit_exclusion_index(original_train: pd.DataFrame) -> ExactInputExclusionIndex:
    """Use only supplied original training labels, never augmented descendants."""
    if LABEL not in original_train:
        raise ValueError(f"Training rows require {LABEL}.")
    label_sets: dict[str, set[str]] = {}
    for key, label in zip(exact_input_keys(original_train), original_train[LABEL]):
        if key:
            label_sets.setdefault(key, set()).add(str(label))
    return ExactInputExclusionIndex(
        {key: tuple(sorted(labels)) for key, labels in label_sets.items()}, len(original_train),
    )


def score_model(model, frame: pd.DataFrame, model_id: str):
    """Return scores and their class order for an evaluated fitted pipeline."""
    inputs = prediction_input(frame, model_id)
    scores = model.predict_log_proba(inputs) if model_id == "NB" else model.decision_function(inputs)
    return np.asarray(scores, dtype=float), np.asarray(model.classes_, dtype=object)
