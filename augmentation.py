"""Frozen conservative augmentation and matched controls for the project.

Only isolated occurrences of the article 'the' are deleted, from transcription
only. This is an EDA-inspired random-deletion adaptation, not clinical
paraphrasing or evidence of new clinical information. No validation/test data
are read here. The same helpers support cross-validation and the final fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import re

import numpy as np
import pandas as pd

LABEL = "medical_specialty"
TEXT_FIELDS = ("sample_name", "transcription")
CONDITIONS = ("A", "B", "C", "D", "E")
CONDITION_NAMES = {
    "A": "LR, no class weights, no augmentation",
    "B": "LR, balanced, no augmentation",
    "C": "LR, no class weights, article dropout",
    "D": "LR, balanced, article dropout",
    "E": "LR, balanced, matched exact copies",
}
# Fixed before scores: the four original training classes with n < 100.
AUGMENTED_CLASSES = (
    "Dermatology", "Neurosurgery", "Ophthalmology", "Psychiatry-Psychology",
)
ARTICLE_RATE = 0.20
LR_C = 0.25
MAX_ITER = 2000
ARTICLE = re.compile(r"(?u)\bthe\b", flags=re.IGNORECASE)
WORDS = re.compile(r"(?u)\b\w+\b")


@dataclass
class DropoutResult:
    text: str
    removed_spans: tuple[tuple[int, int], ...]
    article_count: int
    original_word_count: int
    rejection: str


@dataclass
class TrainingBatch:
    X: pd.DataFrame | pd.Series
    y: pd.Series
    sample_weight: np.ndarray
    audit: pd.DataFrame
    children: pd.DataFrame
    metadata: dict


def normalize_text(value) -> str:
    """The deterministic whitespace cleaning used before TF-IDF."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_inputs(frame: pd.DataFrame, fields=TEXT_FIELDS) -> pd.DataFrame:
    return frame.loc[:, list(fields)].apply(lambda col: col.map(normalize_text))


def identity_groups(train: pd.DataFrame) -> pd.Series:
    """Group repeated notes; title fallback closes empty-transcription overlap.

    Group keys use whitespace normalization and lowercase, matching the input
    cleaning and TF-IDF lowercase. With nonempty transcription, all copies of
    that note stay together regardless of title/label. Otherwise use the title.
    Entirely missing model inputs retain individual groups and remain evaluated.
    These groups do not identify patients or detect merely similar notes.
    """
    X = clean_inputs(train).apply(lambda col: col.str.lower())
    ids = train["record_id"] if "record_id" in train else pd.Series(train.index, index=train.index)
    if ids.duplicated().any():
        raise ValueError("record_id must be unique when creating identity groups.")
    groups = []
    for idx, row in X.iterrows():
        if row["transcription"]:
            payload = ["transcription", row["transcription"]]
        elif row["sample_name"]:
            payload = ["available_fields", row["sample_name"], ""]
        else:
            payload = ["empty_record", str(ids.loc[idx])]
        groups.append(hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest())
    return pd.Series(groups, index=train.index, name="group_id")


def article_dropout(text: str, *, seed: int, fold: int, record_id) -> DropoutResult:
    """Delete max(1, floor(.20*m)) of m isolated 'the' occurrences.

    All other characters survive apart from deterministic whitespace cleanup.
    The seed depends on source ID, never its label. Sources without the article,
    or whose transcription would become empty, are logged and not duplicated.
    """
    text = normalize_text(text)
    matches = list(ARTICLE.finditer(text))
    word_count = len(WORDS.findall(text))
    if not matches:
        return DropoutResult(text, (), 0, word_count, "no_article")
    key = f"{seed}|{fold}|{record_id}|transcription".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
    n_remove = max(1, math.floor(ARTICLE_RATE * len(matches)))
    chosen = sorted(rng.sample(range(len(matches)), n_remove))
    spans = tuple((matches[i].start(), matches[i].end()) for i in chosen)
    pieces, cursor = [], 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    changed = normalize_text("".join(pieces))
    if not changed:
        return DropoutResult(text, (), len(matches), word_count, "empty_after_deletion")
    return DropoutResult(changed, spans, len(matches), word_count, "")


def _augmentation_sources(train: pd.DataFrame, *, seed: int, fold: int):
    source = train.reset_index(drop=True).copy()
    source.loc[:, list(TEXT_FIELDS)] = clean_inputs(source)
    rows, audit = [], []
    for _, row in source.iterrows():
        if row[LABEL] not in AUGMENTED_CLASSES:
            continue
        changed = article_dropout(row["transcription"], seed=seed, fold=fold, record_id=row["record_id"])
        item = {
            "fold": fold, "parent_record_id": row["record_id"], "class": row[LABEL],
            "eligible": True, "changed": bool(changed.removed_spans), "rejection": changed.rejection,
            "article_count": changed.article_count, "removed_words": len(changed.removed_spans),
            "original_word_count": changed.original_word_count,
            "removed_word_fraction": len(changed.removed_spans) / max(1, changed.original_word_count),
            "removed_spans": json.dumps(changed.removed_spans),
        }
        audit.append(item)
        if changed.removed_spans:
            rows.append({
                "record_id": f"aug-{fold}-{row['record_id']}", "parent_record_id": row["record_id"],
                LABEL: row[LABEL], "sample_name": row["sample_name"], "transcription": changed.text,
            })
    columns = ["record_id", "parent_record_id", LABEL, *TEXT_FIELDS]
    audit_columns = [
        "fold", "parent_record_id", "class", "eligible", "changed", "rejection", "article_count",
        "removed_words", "original_word_count", "removed_word_fraction", "removed_spans",
    ]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(audit, columns=audit_columns)


def prepare_training(train: pd.DataFrame, condition: str, *, seed: int = 0, fold: int = 0) -> TrainingBatch:
    """Build one training arm, with total sample-weight mass N_original.

    A: unweighted original data. B: balanced original data. C: unweighted data
    with changed children. D: balanced data with changed children. E: balanced
    data with exact copies of precisely D's accepted parents, in the same order.
    Class weights come only from original training labels and are fixed before
    adding children/copies. The total weight is N_original in every arm, so
    augmentation does not silently weaken regularization by increasing its mass.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    required = {"record_id", LABEL, *TEXT_FIELDS}
    if missing := required - set(train.columns):
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    original = train.reset_index(drop=True).copy()
    if original.empty or original["record_id"].duplicated().any():
        raise ValueError("Training data must be nonempty with unique source IDs.")
    X_original = clean_inputs(original)
    y_original = original[LABEL].astype(str)
    children, audit = _augmentation_sources(original, seed=seed, fold=fold)
    added = children.copy() if condition in {"C", "D"} else children.iloc[:0].copy()
    if condition == "E" and len(children):
        sources = original.set_index("record_id").loc[children["parent_record_id"].tolist()]
        added = pd.DataFrame({
            "record_id": [f"copy-{fold}-{rid}" for rid in children["parent_record_id"]],
            "parent_record_id": children["parent_record_id"].to_numpy(),
            LABEL: sources[LABEL].to_numpy(),
            **{field: sources[field].map(normalize_text).to_numpy() for field in TEXT_FIELDS},
        })
    X = pd.concat([X_original, clean_inputs(added)], ignore_index=True)
    y = pd.concat([y_original, added[LABEL].astype(str)], ignore_index=True)
    balanced = condition in {"B", "D", "E"}
    counts = y_original.value_counts()
    class_weights = {
        label: float(len(y_original) / (len(counts) * count)) if balanced else 1.0
        for label, count in counts.items()
    }
    raw_weights = y.map(class_weights).to_numpy(dtype=float)
    normalizer = len(y_original) / float(raw_weights.sum())
    weights = raw_weights * normalizer
    metadata = {
        "condition": condition, "name": CONDITION_NAMES[condition],
        "original_rows": len(original), "eligible_rows": len(audit), "accepted_sources": len(children),
        "added_rows": len(added), "fit_rows": len(y), "balanced": balanced,
        "class_weights_from_originals": class_weights, "weight_normalizer": normalizer,
        "sample_weight_sum": float(weights.sum()), "sample_weight_mean_over_fit_rows": float(weights.mean()),
        "weight_mass_by_class": {label: float(weights[y.eq(label).to_numpy()].sum()) for label in counts.index},
        "input_fields": list(TEXT_FIELDS), "input_kind": "dataframe",
    }
    return TrainingBatch(X, y, weights, audit, children, metadata)


def build_condition_model():
    """Weights are passed as sample_weight; do not apply class_weight twice."""
    from models import logistic_regression

    return logistic_regression(
        fields=TEXT_FIELDS, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        C=LR_C, class_weight=None, max_iter=MAX_ITER, random_state=0,
    )


def fit_condition(train: pd.DataFrame, condition: str, *, seed: int = 0, fold: int = 0):
    """Fit the frozen LR arm on the supplied original training rows only."""
    from threadpoolctl import threadpool_limits

    batch = prepare_training(train, condition, seed=seed, fold=fold)
    model = build_condition_model()
    with threadpool_limits(limits=1):
        model.fit(batch.X, batch.y, classifier__sample_weight=batch.sample_weight)
    return model, batch


def build_evaluated_model(model_id: str):
    """The seven frozen configurations; no search or test-driven changes."""
    if model_id in CONDITIONS:
        return build_condition_model()
    if model_id == "NB":
        from models import naive_bayes

        return naive_bayes(stem=True)
    if model_id == "SVC":
        from sklearn.svm import LinearSVC

        model = build_condition_model()
        model.set_params(classifier=LinearSVC(
            C=LR_C, class_weight=None, max_iter=MAX_ITER, random_state=0,
        ))
        return model
    raise ValueError("Unknown model_id; use A, B, C, D, E, NB or SVC.")


def prediction_input(frame: pd.DataFrame, model_id: str):
    """Apply deterministic input cleaning, using no learned corpus statistics."""
    if model_id == "NB":
        return clean_inputs(frame, ("description",))["description"]
    if model_id in {*CONDITIONS, "SVC"}:
        return clean_inputs(frame)
    raise ValueError("Unknown model_id; use A, B, C, D, E, NB or SVC.")


def fit_evaluated_model(train: pd.DataFrame, model_id: str, *, seed: int = 0, fold: int = 0):
    """Fit any evaluated family using the exact CV preparation and weighting."""
    from threadpoolctl import threadpool_limits

    base_condition = model_id if model_id in CONDITIONS else "B" if model_id == "SVC" else "A"
    model = build_evaluated_model(model_id)
    batch = prepare_training(train, base_condition, seed=seed, fold=fold)
    batch.metadata["model_id"] = model_id
    if model_id == "NB":
        batch.X = prediction_input(train, model_id).reset_index(drop=True)
        batch.metadata.update({"input_fields": ["description"], "input_kind": "series"})
    with threadpool_limits(limits=1):
        model.fit(batch.X, batch.y, classifier__sample_weight=batch.sample_weight)
    return model, batch
