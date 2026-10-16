"""Within-class section mixtures and matched-copy controls, using originals only.

One complete section body is borrowed from another same-class training note.
This is a synthetic textual mixture, not a coherent real patient record or a
meaning-preserving paraphrase. No validation/test data are read here. The same
preparation supports cross-validation and a later explicitly authorized fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd

LABEL = "medical_specialty"
TEXT_FIELDS = ("sample_name", "transcription")
CONDITIONS = ("A", "B", "C", "D", "E")
EVALUATED_MODEL_IDS = ("NB", "B", "D", "E", "SVC", "SVC_AUG", "SVC_COPY")
AUGMENTATION_NAME = "within-class matched-heading section recombination"
CONDITION_NAMES = {
    "A": "LR, no class weights, no augmentation",
    "B": "LR, balanced, no augmentation",
    "C": "LR, no class weights, section recombination",
    "D": "LR, balanced, section recombination",
    "E": "LR, balanced, matched exact copies",
}
# Fixed before scores: the four original training classes with n < 100.
AUGMENTED_CLASSES = (
    "Dermatology", "Neurosurgery", "Ophthalmology", "Psychiatry-Psychology",
)
LR_C = 0.25
MAX_ITER = 2000
WORDS = re.compile(r"(?u)\b\w+\b")
MIN_SECTION_WORDS = 8
MIN_RECOGNIZED_SECTION_TYPES = 2
# Literal clinical headings observed in the supplied training text. No generic
# uppercase phrase becomes a section, and no semantic donor filter is applied.
SECTION_HEADINGS = (
    "CHIEF COMPLAINT", "CC", "HISTORY OF PRESENT ILLNESS", "HPI", "HISTORY",
    "PAST MEDICAL HISTORY", "PMH", "PAST SURGICAL HISTORY", "PSH",
    "SOCIAL HISTORY", "FAMILY HISTORY", "REVIEW OF SYSTEMS", "ROS",
    "MEDICATIONS", "MEDS", "ALLERGIES", "PHYSICAL EXAMINATION", "PHYSICAL EXAM",
    "EXAMINATION", "EXAM", "VITAL SIGNS", "GENERAL", "HEENT", "NECK", "HEART",
    "LUNGS", "CHEST", "ABDOMEN", "EXTREMITIES", "NEUROLOGIC", "NEUROLOGICAL",
    "PREOPERATIVE DIAGNOSIS", "PREOPERATIVE DIAGNOSES", "POSTOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSES", "PROCEDURE", "PROCEDURES", "PROCEDURE PERFORMED",
    "PROCEDURES PERFORMED", "OPERATIVE PROCEDURE", "OPERATIONS PERFORMED",
    "OPERATION PERFORMED", "TITLE OF OPERATION", "TITLE OF THE OPERATION",
    "DESCRIPTION OF PROCEDURE", "PROCEDURE IN DETAIL", "OPERATIVE FINDINGS",
    "FINDINGS", "INDICATION", "INDICATIONS", "INDICATIONS FOR PROCEDURE",
    "ANESTHESIA", "ESTIMATED BLOOD LOSS", "COMPLICATIONS", "SPECIMENS", "DRAINS",
    "DIAGNOSIS", "DIAGNOSES", "ASSESSMENT", "PLAN", "ASSESSMENT AND PLAN",
    "IMPRESSION", "RECOMMENDATIONS", "DISPOSITION", "DISCHARGE DIAGNOSIS",
    "DISCHARGE DIAGNOSES", "HOSPITAL COURSE", "TECHNIQUE", "CONCLUSION", "CONCLUSIONS",
    "SUBJECTIVE", "OBJECTIVE", "TREATMENT", "CURRENT MEDICATIONS", "SKIN",
    "FAMILY, SOCIAL, AND ALLERGY HISTORY", "EMERGENCY DEPARTMENT COURSE",
    "PROCEDURE NOTE", "TITLE OF PROCEDURES", "PATHOLOGY", "BLOOD LOSS", "FLUIDS",
    "COUNTS", "IMMUNIZATIONS", "FINAL DIAGNOSIS", "FINAL DIAGNOSES",
    "HX", "COURSE", "MS", "CN", "REASON FOR EXAM", "CARDIOVASCULAR", "FHX", "SHX",
    "LABORATORY DATA", "SPECIMEN", "OPERATION", "TOURNIQUET TIME", "BRIEF HISTORY",
    "MUSCULOSKELETAL", "DESCRIPTION OF OPERATION", "INTERPRETATION", "PREOP DIAGNOSIS",
    "RESPIRATORY", "GROSS FINDINGS", "SUMMARY", "URINE OUTPUT", "PROCEDURE DETAIL",
    "INDICATION FOR SURGERY", "MOTOR", "CONDITION", "POSTOP DIAGNOSIS",
    "INTRAOPERATIVE FINDINGS", "REASON FOR CONSULTATION", "PROCEDURE DETAILS",
    "OPERATIVE PROCEDURE IN DETAIL", "CLINICAL", "DISCUSSION", "REASON FOR VISIT",
    "DESCRIPTION OF THE PROCEDURE", "GASTROINTESTINAL", "GU", "INDICATIONS FOR SURGERY",
    "CLINICAL HISTORY", "GENITOURINARY", "PSYCHIATRIC", "INDICATIONS FOR OPERATION",
    "SENSORY", "GI", "DISCHARGE MEDICATIONS", "DISCHARGE INSTRUCTIONS", "HEMOSTASIS",
    "RECTAL", "RECOMMENDATION", "NAME OF OPERATION", "HISTORY AND PHYSICAL",
    "DETAILS OF THE PROCEDURE", "EYES", "BACK", "DELIVERY NOTE", "OPERATIONS",
    "GEN EXAM", "CNS", "HEMODYNAMIC DATA", "CARDIAC", "PREOPERATIVE DIAGNOSIS (ES)",
    "OPERATIVE TECHNIQUE", "PAST PSYCHIATRIC HISTORY", "REASON FOR REFERRAL",
    "PERSONAL HISTORY", "DIAGNOSTIC DATA", "RIGHT CORONARY ARTERY", "INFORMED CONSENT",
    "ENDOCRINE", "NARRATIVE", "LABORATORY STUDIES", "POSTOPERATIVE DIAGNOSIS (ES)",
    "DESCRIPTION OF THE OPERATION", "PREPROCEDURE DIAGNOSIS", "POSTPROCEDURE DIAGNOSIS",
    "INDICATION FOR PROCEDURE", "DESCRIPTION", "OPERATION IN DETAIL", "IMPRESSION AND PLAN",
    "SHX/FHX", "TOTAL TOURNIQUET TIME", "POSTOPERATIVE PLAN", "HISTORY AND GROSS FINDINGS",
)
_HEADINGS = "|".join(re.escape(s).replace(r"\ ", r"[ \t]+") for s in sorted(SECTION_HEADINGS, key=lambda s: (-len(s), s)))
HEADING_PATTERN = re.compile(r"(?:(?<=^)|(?<=[,:\r\n]))[ \t]*(?P<heading>" + _HEADINGS + r")[ \t]*:")
# An unrecognized uppercase heading is not promoted to a valid section type.
# Its presence inside an apparent body makes that body structurally ambiguous.
UNRESOLVED_HEADING_PATTERN = re.compile(r"(?:(?<=^)|(?<=[,:\r\n]))[ \t]*[A-Z][A-Z /&()\-]{1,65}:")


@dataclass(frozen=True)
class ClinicalSection:
    section_type: str
    heading_start: int
    body_start: int
    body_end: int
    body: str
    word_count: int


@dataclass(frozen=True)
class SectionMixResult:
    text: str
    donor_record_id: object
    section_type: str
    parent_body_span: tuple[int, int] | None
    donor_body_span: tuple[int, int] | None
    original_word_count: int
    augmented_word_count: int
    removed_words: int
    added_words: int
    parent_section_types: int
    donor_section_types: int
    candidate_pairs: int
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


def parse_sections(text: str) -> tuple[ClinicalSection, ...]:
    """Recognize allowlisted uppercase headings at note/comma/line boundaries.

    Offsets refer to the original transcription, before whitespace cleaning.
    Only content between complete recognized headings is a section body; no
    sentences, dosages or numeric values are cut to impose a length budget.
    """
    text = "" if text is None or pd.isna(text) else str(text)
    matches = list(HEADING_PATTERN.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start("heading") if i + 1 < len(matches) else len(text)
        while start < end and text[start].isspace():
            start += 1
        # The dataset encodes heading boundaries with commas as well as newlines.
        # Preserve the recipient's boundary separators outside the replaced body.
        while end > start and (text[end - 1].isspace() or text[end - 1] == ","):
            end -= 1
        body = text[start:end]
        sections.append(ClinicalSection(
            section_type=" ".join(match.group("heading").split()),
            heading_start=match.start("heading"), body_start=start, body_end=end,
            body=body, word_count=len(WORDS.findall(body)),
        ))
    return tuple(sections)


def _section_types(sections):
    return len({section.section_type for section in sections if section.word_count})


def _usable_section(section, sections):
    return (section.word_count >= MIN_SECTION_WORDS
            and sum(s.section_type == section.section_type for s in sections) == 1
            and not UNRESOLVED_HEADING_PATTERN.search(section.body))


def _id_order(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _mix_record(parent, sections, groups, by_class, *, seed, fold):
    rid = parent["record_id"]
    text = str(parent["transcription"])
    words = len(WORDS.findall(text))
    own = sections[rid]
    own_types = _section_types(own)

    def rejected(reason):
        return SectionMixResult(text, "", "", None, None, words, words, 0, 0, own_types, 0, 0, reason)

    if not text.strip():
        return rejected("empty_transcription")
    if own_types < MIN_RECOGNIZED_SECTION_TYPES:
        return rejected("insufficient_recognized_section_types")
    replaceable = [section for section in own if _usable_section(section, own)]
    if not replaceable:
        return rejected("no_unambiguous_parent_section_at_min_words")
    donors = [row for row in by_class[parent[LABEL]]
              if row["record_id"] != rid and groups[row["record_id"]] != groups[rid]
              and _section_types(sections[row["record_id"]]) >= MIN_RECOGNIZED_SECTION_TYPES]
    if not donors:
        return rejected("no_distinct_same_class_original_donor")
    pairs = []
    for section in replaceable:
        for donor in donors:
            for borrowed in sections[donor["record_id"]]:
                if (borrowed.section_type == section.section_type
                        and _usable_section(borrowed, sections[donor["record_id"]])
                        and normalize_text(borrowed.body).lower() != normalize_text(section.body).lower()):
                    pairs.append((section, donor, borrowed))
    if not pairs:
        return rejected("no_matching_different_section_at_min_words")
    pairs.sort(key=lambda pair: (pair[0].section_type, _id_order(pair[1]["record_id"]), pair[0].body_start, pair[2].body_start))
    key = f"{seed}|{fold}|{rid}|section-recombination".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
    section, donor, borrowed = pairs[rng.randrange(len(pairs))]
    mixed = text[:section.body_start] + borrowed.body + text[section.body_end:]
    return SectionMixResult(
        mixed, donor["record_id"], section.section_type,
        (section.body_start, section.body_end), (borrowed.body_start, borrowed.body_end),
        words, len(WORDS.findall(mixed)), section.word_count, borrowed.word_count,
        own_types, _section_types(sections[donor["record_id"]]), len(pairs), "",
    )


def _augmentation_sources(train: pd.DataFrame, *, seed: int, fold: int):
    """Borrow solely from the supplied original training frame, never children."""
    if "parent_record_id" in train and train["parent_record_id"].fillna("").astype(str).ne("").any():
        raise ValueError("Only original training records may be recipients or donors.")
    source = train.reset_index(drop=True).copy()
    targeted = source.loc[source[LABEL].isin(AUGMENTED_CLASSES)]
    records = targeted.to_dict("records")
    sections = {row["record_id"]: parse_sections(row["transcription"]) for row in records}
    group_series = identity_groups(source)
    groups = dict(zip(source["record_id"], group_series))
    by_class = {label: sorted([r for r in records if r[LABEL] == label], key=lambda r: _id_order(r["record_id"]))
                for label in AUGMENTED_CLASSES}
    rows, audit = [], []
    for row in records:
        mixed = _mix_record(row, sections, groups, by_class, seed=seed, fold=fold)
        accepted = not mixed.rejection
        audit.append({
            "fold": fold, "parent_record_id": row["record_id"], "donor_record_id": mixed.donor_record_id,
            "class": row[LABEL], "section_type": mixed.section_type, "eligible": True,
            "changed": accepted, "rejection": mixed.rejection,
            "parent_group_id": groups[row["record_id"]],
            "donor_group_id": groups[mixed.donor_record_id] if accepted else "",
            "parent_section_types": mixed.parent_section_types, "donor_section_types": mixed.donor_section_types,
            "candidate_pairs": mixed.candidate_pairs, "original_word_count": mixed.original_word_count,
            "augmented_word_count": mixed.augmented_word_count, "removed_words": mixed.removed_words,
            "added_words": mixed.added_words,
            "removed_word_fraction": mixed.removed_words / max(1, mixed.original_word_count),
            "added_word_fraction": mixed.added_words / max(1, mixed.original_word_count),
            "parent_body_span": json.dumps(mixed.parent_body_span), "donor_body_span": json.dumps(mixed.donor_body_span),
        })
        if accepted:
            rows.append({
                "record_id": f"aug-{fold}-{row['record_id']}", "parent_record_id": row["record_id"],
                "donor_record_id": mixed.donor_record_id, "section_type": mixed.section_type,
                LABEL: row[LABEL], "sample_name": row["sample_name"], "transcription": mixed.text,
            })
    columns = ["record_id", "parent_record_id", "donor_record_id", "section_type", LABEL, *TEXT_FIELDS]
    audit_columns = [
        "fold", "parent_record_id", "donor_record_id", "class", "section_type", "eligible", "changed", "rejection",
        "parent_group_id", "donor_group_id", "parent_section_types", "donor_section_types", "candidate_pairs",
        "original_word_count", "augmented_word_count", "removed_words", "added_words", "removed_word_fraction",
        "added_word_fraction", "parent_body_span", "donor_body_span",
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
        "augmentation_name": AUGMENTATION_NAME,
        "augmentation_interpretation": "synthetic same-specialty section mixture; not a coherent real patient",
        "no_article_deletion": True, "no_markers_in_model_input": True,
        "min_section_words": MIN_SECTION_WORDS,
        "min_recognized_section_types": MIN_RECOGNIZED_SECTION_TYPES,
        "skip_repeated_or_unresolved_heading_spans": True,
        "original_rows": len(original), "eligible_rows": len(audit), "accepted_sources": len(children),
        "added_rows": len(added), "fit_rows": len(y), "balanced": balanced,
        "class_weights_from_originals": class_weights, "weight_normalizer": normalizer,
        "sample_weight_sum": float(weights.sum()), "sample_weight_mean_over_fit_rows": float(weights.mean()),
        "weight_mass_by_class": {label: float(weights[y.eq(label).to_numpy()].sum()) for label in counts.index},
        "input_fields": list(TEXT_FIELDS), "input_kind": "dataframe",
        "accepted_sources_by_class": children[LABEL].value_counts().to_dict(),
        "rejection_counts": audit.loc[~audit["changed"], "rejection"].value_counts().to_dict(),
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
    """Construct the registered model without fitting or selecting it."""
    if model_id in CONDITIONS:
        return build_condition_model()
    if model_id == "NB":
        from models import naive_bayes

        return naive_bayes(stem=True)
    if model_id in {"SVC", "SVC_AUG", "SVC_COPY"}:
        from sklearn.svm import LinearSVC

        model = build_condition_model()
        model.set_params(classifier=LinearSVC(
            C=LR_C, class_weight=None, max_iter=MAX_ITER, random_state=0,
        ))
        return model
    raise ValueError("Unknown model_id; use A-E, NB, SVC, SVC_AUG or SVC_COPY.")


def prediction_input(frame: pd.DataFrame, model_id: str):
    """Apply deterministic input cleaning, using no learned corpus statistics."""
    if model_id == "NB":
        return clean_inputs(frame, ("description",))["description"]
    if model_id in {*CONDITIONS, "SVC", "SVC_AUG", "SVC_COPY"}:
        return clean_inputs(frame)
    raise ValueError("Unknown model_id; use A-E, NB, SVC, SVC_AUG or SVC_COPY.")


def fit_evaluated_model(train: pd.DataFrame, model_id: str, *, seed: int = 0, fold: int = 0):
    """Fit any evaluated family using the exact CV preparation and weighting."""
    from threadpoolctl import threadpool_limits

    base_condition = (model_id if model_id in CONDITIONS else
                      {"SVC": "B", "SVC_AUG": "D", "SVC_COPY": "E", "NB": "A"}.get(model_id))
    if base_condition is None:
        raise ValueError("Unknown evaluated model ID.")
    model = build_evaluated_model(model_id)
    batch = prepare_training(train, base_condition, seed=seed, fold=fold)
    batch.metadata["model_id"] = model_id
    if model_id == "NB":
        batch.X = prediction_input(train, model_id).reset_index(drop=True)
        batch.metadata.update({"input_fields": ["description"], "input_kind": "series"})
    with threadpool_limits(limits=1):
        model.fit(batch.X, batch.y, classifier__sample_weight=batch.sample_weight)
    return model, batch
