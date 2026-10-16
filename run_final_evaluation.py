r"""Evaluate seven base systems and a shared exact-input decoder, train data only.

    .venv\Scripts\python.exe run_final_evaluation.py

The primary scenario splits rows; grouped notes are a separate robustness
scenario. Both are exploratory development evaluations after previous audits.
No official test, selected model, or results.txt is read or modified.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from augmentation import (
    AUGMENTATION_NAME, AUGMENTED_CLASSES, EVALUATED_MODEL_IDS, LABEL, LR_C,
    MAX_ITER, MIN_SECTION_WORDS, MIN_RECOGNIZED_SECTION_TYPES, SECTION_HEADINGS, TEXT_FIELDS,
    clean_inputs, fit_evaluated_model, identity_groups, prediction_input,
)
from augmentation_policy import (
    MODEL_IDS, MODEL_NAMES, PROTOCOLS, DECODERS, SOURCE_FILES,
    POLICY_SPEC, SELECTION_RULE, choose_candidate,
)
from config import FINAL_EVALUATION_DIR
from data_preparation import repair_training_csv, sha256
from duplicate_exclusion import exact_input_keys, fit_exclusion_index, score_model

ROOT = Path(__file__).resolve().parent
OUT = FINAL_EVALUATION_DIR
SEED = 0
N_FOLDS = 5
GENERATED: set[Path] = set()
ACTIVE_RUN = False


def candidate_id(model_id, decoder):
    return model_id if decoder == "raw" else f"{model_id}+EXCL"


def save_json(path, value, *, artifact=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if artifact:
        GENERATED.add(path)


def write_csv(frame, path, *, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=";", encoding="utf-8-sig", index=index)
    GENERATED.add(path)


def metrics(actual, predicted, classes):
    report = classification_report(actual, predicted, labels=classes, output_dict=True, zero_division=0)
    worst = min(classes, key=lambda label: report[label]["f1-score"])
    accuracy = float(accuracy_score(actual, predicted))
    minimum = float(report[worst]["f1-score"])
    return {
        "accuracy": accuracy, "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]), "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]), "min_f1": minimum, "worst_class": worst,
        "zero_f1_classes": sum(report[label]["f1-score"] == 0 for label in classes),
        "meets_criteria": accuracy > .49 and minimum >= .25,
    }, report


def transition_counts(frame):
    return {
        "n": len(frame), "has_train_match": int(frame.has_train_match.sum()),
        "title_only_matches": int((frame.has_train_match & ~frame.nonempty_transcription).sum()),
        "changed": int(frame.changed.sum()), "fallback_all_excluded": int(frame.fallback_all_excluded.sum()),
        "corrected": int((~frame.raw_correct & frame.correct).sum()),
        "harmed": int((frame.raw_correct & ~frame.correct).sum()),
        "changed_still_wrong": int((frame.changed & ~frame.raw_correct & ~frame.correct).sum()),
        "true_class_seen_in_training_matches": int(frame.true_class_seen_in_training_matches.sum()),
    }



def run():
    global ACTIVE_RUN
    if tuple(EVALUATED_MODEL_IDS) != MODEL_IDS:
        raise ValueError("Augmentation model IDs differ from the frozen promotion policy.")
    GENERATED.clear()
    if (OUT / "experiment.json").exists() or (OUT / "run_state.json").exists():
        raise FileExistsError("Preserve the existing run; use a new output directory rather than overwriting it.")
    ACTIVE_RUN = True
    OUT.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    source_names = SOURCE_FILES
    hashes = {name: sha256(ROOT / name) for name in source_names}
    records, _ = repair_training_csv(ROOT / "train.csv")
    train = pd.DataFrame(records).reset_index(drop=True)
    classes = sorted(train[LABEL].unique())
    groups = identity_groups(train)
    design = {
        "status": "post-hoc exploratory development comparison; not independent post-selection or official-test evaluation",
        "seed": SEED, "n_folds": N_FOLDS, "LR_C": LR_C,
        "protocols": list(PROTOCOLS), "primary_selection_protocol": PROTOCOLS[0],
        "model_ids": list(MODEL_IDS), "decoders": list(DECODERS),
        "candidate_ids": [candidate_id(mid, dec) for mid in MODEL_IDS for dec in DECODERS],
        "source_sha256": hashes["train.csv"], "sources_sha256": hashes,
        "original_records": len(train), "classes": classes, "class_counts": train[LABEL].value_counts().to_dict(),
        "tfidf": {"fields": list(TEXT_FIELDS), "per_field": True, "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True,
                  "fit_scope": "only each arm's original training rows and added training descendants"},
        "models": {"LR": {"C": LR_C, "max_iter": MAX_ITER, "loss": "multinomial", "penalty": "L2"},
                   "SVC": {"C": LR_C, "max_iter": MAX_ITER, "loss": "squared_hinge", "multiclass": "ovr"},
                   "NB": {"fields": ["description"], "Porter_stemming": True, "alpha": 1.0, "fit_prior": True}},
        "augmentation": {"name": AUGMENTATION_NAME, "classes": list(AUGMENTED_CLASSES), "field": "transcription",
                         "rule": "replace one matched-heading section body using another original same-class training note from a different identity group",
                         "minimum_section_words": MIN_SECTION_WORDS,
                         "minimum_recognized_section_types": MIN_RECOGNIZED_SECTION_TYPES,
                         "allowed_headings": list(SECTION_HEADINGS), "variants_per_accepted_parent": 1,
                         "ambiguous_sections": "skip repeated heading types and bodies with unresolved uppercase headings in either source",
                         "same_patient_semantics_claimed": False},
        "conditions": {"B": "LR balanced, no augmentation", "D": "LR balanced section recombination",
                       "E": "LR balanced matched copies", "NB": "specified description baseline",
                       "SVC": "SVC balanced, no augmentation", "SVC_AUG": "SVC balanced section recombination",
                       "SVC_COPY": "SVC balanced matched copies"},
        "weights": "balanced counts estimated on original fold training rows, then total sample-weight mass normalized to N_original; D/E/SVC_AUG/SVC_COPY match parents, counts and weights",
        "decoder": {"key": "normalized lowercase (sample_name,transcription), skip only both-empty",
                    "index_scope": "only original rows of the training fold, never augmented descendants or validation targets",
                    "operation": "exclude all observed classes; next-highest score; raw fallback for no match or all excluded",
                    "scores": "NB predict_log_proba, other models decision_function",
                    "premise": "Exact input copies tend to have distinct categories in this corpus; a dataset-construction artifact, not a clinical law.",
                    "post_hoc": True},
        "group_rule": "original identity_groups: normalized nonempty transcription, title fallback, individual groups for wholly empty inputs",
        "selection": SELECTION_RULE, "selection_policy": POLICY_SPEC,
        "limitations": ["Field, parameter, augmentation and decoder choices used this labelled corpus.",
                        "Raw, decoded, row and grouped outcomes answer different evaluation conditions.",
                        "No test labels are available; no official-test performance is measured.",
                        "A single conservative augmentation and fixed family settings do not rank all methods."],
    }
    save_json(OUT / "design.json", design)
    save_json(OUT / "run_state.json", {"state": "running", "started_utc": start_utc}, artifact=False)
    summaries, per_class, fold_metrics, predictions, weight_metadata, audit_frames, augmentation_examples, decoder_diagnostics, warning_log = [], [], [], [], [], [], [], [], []
    all_wide, all_assignments = [], []
    print("Design saved before fits. Primary=stratified_rows_development; grouped is robustness only.", flush=True)
    for protocol in PROTOCOLS:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED) if protocol == PROTOCOLS[0] else StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = list(splitter.split(train, train[LABEL], groups)) if protocol == PROTOCOLS[1] else list(splitter.split(train, train[LABEL]))
        assignments = np.zeros(len(train), dtype=int)
        protocol_frames = []
        for fold, (tr, va) in enumerate(folds, 1):
            assert not (set(tr) & set(va))
            if protocol == PROTOCOLS[1]:
                assert not (set(groups.iloc[tr]) & set(groups.iloc[va]))
            assignments[va] = fold
            training, validation = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
            assert set(training[LABEL]) == set(classes)
            lookup = fit_exclusion_index(training)
            matched_batches = {}
            for model_id in MODEL_IDS:
                fitted_at = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model, batch = fit_evaluated_model(training, model_id, seed=SEED, fold=fold)
                    score_values, score_classes = score_model(model, validation, model_id)
                for item in caught:
                    warning_log.append({"protocol": protocol, "fold": fold, "model_id": model_id, "warning": str(item.message)})
                decoded = lookup.predict(score_values, score_classes, validation)
                assert np.array_equal(decoded.raw_predictions, model.predict(prediction_input(validation, model_id)))
                if protocol == PROTOCOLS[1]:
                    assert not decoded.diagnostics.changed.any()
                    assert not decoded.diagnostics.has_train_match.any()
                weight_metadata.append({"protocol": protocol, "fold": fold, "model_id": model_id, **batch.metadata})
                if model_id in {"D", "E", "SVC_AUG", "SVC_COPY"}:
                    fingerprint = (batch.children.parent_record_id.tolist(), batch.y.tolist(), batch.sample_weight.tolist())
                    matched_batches[model_id] = fingerprint
                    if model_id != "D":
                        assert fingerprint == matched_batches["D"], "Copy/augmentation budgets or weights differ."
                if model_id == "D":
                    audit = batch.audit.copy()
                    audit.insert(0, "protocol", protocol)
                    assert set(batch.children.parent_record_id) <= set(training.record_id)
                    assert set(batch.children.parent_record_id).isdisjoint(validation.record_id)
                    assert set(batch.children.donor_record_id) <= set(training.record_id)
                    assert set(batch.children.donor_record_id).isdisjoint(validation.record_id)
                    audit_frames.append(audit)
                    for _, child in batch.children.groupby(LABEL, sort=True).head(2).iterrows():
                        parent = training.set_index("record_id").loc[child.parent_record_id]
                        augmentation_examples.append({"protocol": protocol, "fold": fold, "parent_record_id": child.parent_record_id,
                                                      "donor_record_id": child.donor_record_id, "section_type": child.section_type,
                                                      "class": child[LABEL], "sample_name": parent.sample_name,
                                                      "original_transcription": parent.transcription,
                                                      "augmented_transcription": child.transcription})
                for decoder, predicted in [("raw", decoded.raw_predictions), ("exclude_seen", decoded.predictions)]:
                    cid = candidate_id(model_id, decoder)
                    detail = decoded.diagnostics.copy()
                    detail["decoder_would_change"] = detail.changed
                    if decoder == "raw":
                        detail["changed"] = False
                    detail["decoder_enabled"] = decoder == "exclude_seen"
                    detail["protocol"], detail["seed"], detail["fold"] = protocol, SEED, fold
                    detail["candidate_id"], detail["model_id"], detail["decoder"] = cid, model_id, decoder
                    detail["record_id"] = validation.record_id.to_numpy()
                    detail["true_class"] = validation[LABEL].to_numpy()
                    detail["predicted_class"], detail["raw_prediction"] = predicted, decoded.raw_predictions
                    detail["group_id"] = groups.iloc[va].to_numpy()
                    detail["nonempty_transcription"] = clean_inputs(validation).transcription.ne("").to_numpy()
                    detail["raw_correct"] = decoded.raw_predictions == validation[LABEL].to_numpy()
                    detail["correct"] = predicted == validation[LABEL].to_numpy()
                    detail["true_class_seen_in_training_matches"] = [
                        str(label) in lookup.label_sets.get(key, ()) if key else False
                        for key, label in zip(exact_input_keys(validation), validation[LABEL])
                    ]
                    protocol_frames.append(detail)
                    values, _ = metrics(detail.true_class, detail.predicted_class, classes)
                    fold_metrics.append({"protocol": protocol, "seed": SEED, "fold": fold, "candidate_id": cid,
                                         "model_id": model_id, "decoder": decoder, "training_originals": len(tr), "validation_originals": len(va),
                                         "fit_rows": len(batch.y), "seconds_base_fit_both_decoders": time.perf_counter() - fitted_at, **values})
                print(f"{protocol} fold{fold}/{N_FOLDS} {model_id}: raw={np.mean(decoded.raw_predictions == validation[LABEL].to_numpy()):.2%} EXCL={np.mean(decoded.predictions == validation[LABEL].to_numpy()):.2%}", flush=True)
                save_json(OUT / "run_state.json", {"state": "running", "started_utc": start_utc, "protocol": protocol,
                                                    "last_completed_fold": fold, "last_completed_model": model_id}, artifact=False)
            write_csv(pd.DataFrame(fold_metrics), OUT / "metrics_per_fold.csv")
        combined = pd.concat(protocol_frames, ignore_index=True)
        wide = train[["record_id", LABEL]].rename(columns={LABEL: "true_class"}).copy()
        wide.insert(0, "protocol", protocol)
        wide.insert(1, "seed", SEED)
        wide["fold"], wide["group_id"] = assignments, groups
        all_assignments.append(wide.drop(columns="true_class").copy())
        write_csv(pd.concat(all_assignments, ignore_index=True), OUT / "fold_assignments.csv")
        protocol_dir = OUT / protocol
        for cid, frame in combined.groupby("candidate_id", sort=False):
            assert len(frame) == len(train) and frame.record_id.nunique() == len(train)
            mid, decoder = frame.model_id.iloc[0], frame.decoder.iloc[0]
            family = MODEL_NAMES[mid]
            values, report = metrics(frame.true_class, frame.predicted_class, classes)
            summaries.append({"candidate_id": cid, "model_id": mid, "decoder": decoder, "model": family,
                              "protocol": protocol, "seed": SEED, **values})
            decoder_diagnostics.append({"protocol": protocol, "candidate_id": cid, "model_id": mid, "decoder": decoder, **transition_counts(frame)})
            for label in classes:
                per_class.append({"protocol": protocol, "seed": SEED, "candidate_id": cid, "model_id": mid, "decoder": decoder,
                                  "class": label, "precision": report[label]["precision"], "recall": report[label]["recall"],
                                  "f1": report[label]["f1-score"], "support": int(report[label]["support"])})
            by_id = frame.set_index("record_id")
            wide[cid] = wide.record_id.map(by_id.predicted_class)
            cm = pd.DataFrame(confusion_matrix(frame.true_class, frame.predicted_class, labels=classes), index=classes, columns=classes).rename_axis("true_class")
            write_csv(cm, protocol_dir / f"confusion_{cid}.csv", index=True)
            write_csv(cm.div(cm.sum(axis=1), axis=0), protocol_dir / f"confusion_normalized_{cid}.csv", index=True)
            errors = frame.loc[~frame.correct].merge(train[["record_id", "description", "sample_name", "transcription"]], on="record_id", how="left", validate="one_to_one")
            write_csv(errors, protocol_dir / f"errors_{cid}.csv")
            examples = []
            pairs = errors.groupby(["true_class", "predicted_class"]).size().sort_values(ascending=False)
            for (actual, pred), count in pairs.head(3).items():
                examples.append({**errors.loc[errors.true_class.eq(actual) & errors.predicted_class.eq(pred)].iloc[0].to_dict(), "confusion_pair_count": int(count)})
            write_csv(pd.DataFrame(examples), protocol_dir / f"error_examples_{cid}.csv")
        predictions.append(combined)
        all_wide.append(wide)
        write_csv(pd.DataFrame(summaries), OUT / "summary.csv")
        write_csv(pd.DataFrame(per_class), OUT / "metrics_per_class.csv")
        write_csv(pd.concat(predictions, ignore_index=True), OUT / "oof_predictions.csv")
        write_csv(pd.concat(all_wide, ignore_index=True), OUT / "oof_predictions_wide.csv")
        write_csv(pd.DataFrame(decoder_diagnostics), OUT / "decoder_diagnostics.csv")
        print(pd.DataFrame(summaries).query("protocol == @protocol").to_string(index=False), flush=True)
    assert len(summaries) == 28
    write_csv(pd.concat(audit_frames, ignore_index=True), OUT / "augmentation_audit.csv")
    write_csv(pd.DataFrame(augmentation_examples), OUT / "augmentation_examples.csv")
    save_json(OUT / "weights_per_fold.json", weight_metadata)
    write_csv(pd.DataFrame(warning_log, columns=["protocol", "fold", "model_id", "warning"]), OUT / "warnings.csv")
    selection = choose_candidate(pd.DataFrame(summaries), pd.DataFrame(per_class))
    write_csv(pd.DataFrame(selection["promotion_diagnostics"]), OUT / "gate_diagnostics.csv")
    save_json(OUT / "promotion_decision.json", selection)
    changed_sources = {name: {"start": hashes[name], "now": sha256(ROOT / name)} for name in source_names if sha256(ROOT / name) != hashes[name]}
    if changed_sources:
        save_json(OUT / "run_state.json", {"state": "failed", "reason": "source hashes changed during evaluation", "changed_sources": changed_sources}, artifact=False)
        raise RuntimeError("Sources changed during evaluation; outputs must not be selected as final.")
    artifacts = {path.relative_to(OUT).as_posix(): sha256(path) for path in sorted(GENERATED)}
    experiment = {
        **design, "state": "complete", "started_utc": start_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "versions": {"python": sys.version.split()[0], **{p: metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "nltk", "threadpoolctl")}},
        "artifacts_sha256": artifacts, "warnings": warning_log,
        "promotion_decision": selection,
        "stability_diagnostics": {"state": "not_reused", "reason": "augmentation source changed; old studies remain archived"},
        "sources_unchanged_at_completion": True,
        "grouped_decoder_inactive": True, "metrics_source": "pooled OOF predictions of all original cases",
    }
    save_json(OUT / "experiment.json", experiment, artifact=False)
    save_json(OUT / "run_state.json", {"state": "complete", "completed_utc": experiment["completed_utc"]}, artifact=False)
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        if ACTIVE_RUN:
            save_json(OUT / "run_state.json", {"state": "failed", "error": str(error)}, artifact=False)
        raise
