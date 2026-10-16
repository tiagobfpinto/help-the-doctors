"""Select using labelled row development; report unseen-note robustness separately.

The official test is opened only after selection, fitting on all originals and
building the original-training exclusion index. Scores are exploratory
development estimates, not independent post-selection or official-test results.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from augmentation import LR_C, fit_evaluated_model
from config import (
    FINAL_EVALUATION_DIR, FINAL_OUTPUT_DIR, MIN_ACCURACY, MIN_CLASS_F1,
    N_FOLDS, RESULTS_FILE, ROOT, SEED, SELECTED_MODEL_FILE, TEST_FILE, TRAIN_FILE,
)
from data_preparation import repair_training_csv, sha256
from duplicate_exclusion import fit_exclusion_index, score_model
from read_test_set import read_test_cases, write_results

MODEL_IDS = {"A", "B", "C", "D", "E", "NB", "SVC"}
MODEL_NAMES = {name: "MultinomialNB" if name == "NB" else "LinearSVC" if name == "SVC"
               else "LogisticRegression" for name in MODEL_IDS}
DECODERS = {"raw", "exclude_seen"}
PRIMARY_PROTOCOL = "stratified_rows_development"
GROUPED_PROTOCOL = "grouped_development"
CANDIDATE_IDS = MODEL_IDS | {name + "+EXCL" for name in MODEL_IDS}
IMPLEMENTATION_FILES = (
    "augmentation.py", "models.py", "data_preparation.py", "duplicate_exclusion.py",
)
SELECTION_RULE = (
    "Use only seed-0 five-fold stratified-row development metrics. Among all "
    "fourteen classifier/decoder candidates with minimum per-class F1 >= 0.25, "
    "maximize accuracy, then macro F1, then minimum F1, then alphabetical "
    "candidate ID. If none satisfy the F1 constraint, maximize macro F1, then "
    "accuracy and minimum F1. Report the same candidate's grouped metrics; "
    "never select the larger score across protocols."
)
EVALUATION_STATUS = (
    "Exploratory development selection after corpus/annotation inspection and "
    "external methodological review; neither row nor grouped CV is an "
    "independent post-selection test evaluation. Exclusion assumes that an "
    "identical input's next annotation tends to use a previously unseen label."
)


def choose_candidate(summary: pd.DataFrame) -> dict:
    """Select in one scenario, checking complete matched reporting in both."""
    required = {"candidate_id", "model_id", "decoder", "model", "protocol", "seed",
                "accuracy", "f1_macro", "min_f1"}
    if missing := required - set(summary.columns):
        raise ValueError(f"Missing experiment columns: {sorted(missing)}")
    protocols = {PRIMARY_PROTOCOL, GROUPED_PROTOCOL}
    if set(summary["protocol"]) != protocols or set(summary["seed"]) != {SEED}:
        raise ValueError("Require exactly the declared row/grouped protocols and seed.")
    if summary.duplicated(["protocol", "candidate_id"]).any():
        raise ValueError("Expected one row per candidate and protocol.")
    for protocol in protocols:
        if set(summary.loc[summary.protocol.eq(protocol), "candidate_id"]) != CANDIDATE_IDS:
            raise ValueError("Run all fourteen candidates in both scenarios before selection.")
    for row in summary.itertuples():
        expected_id = row.model_id + ("+EXCL" if row.decoder == "exclude_seen" else "")
        if (row.model_id not in MODEL_IDS or row.decoder not in DECODERS
                or row.candidate_id != expected_id or row.model != MODEL_NAMES[row.model_id]):
            raise ValueError("Candidate ID, classifier and decoder are inconsistent.")
    scores = summary[["accuracy", "f1_macro", "min_f1"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Metrics must be finite fractions between zero and one.")
    primary = summary.loc[summary.protocol.eq(PRIMARY_PROTOCOL)]
    eligible = primary.loc[primary.min_f1.ge(MIN_CLASS_F1)]
    if len(eligible):
        order = ["accuracy", "f1_macro", "min_f1", "candidate_id"]
    else:
        eligible = primary
        order = ["f1_macro", "accuracy", "min_f1", "candidate_id"]
    row = eligible.sort_values(order, ascending=[False, False, False, True]).iloc[0]
    grouped = summary.loc[summary.protocol.eq(GROUPED_PROTOCOL) & summary.candidate_id.eq(row.candidate_id)].iloc[0]
    metrics = lambda record: {key: float(record[key]) for key in ("accuracy", "f1_macro", "min_f1")}
    return {
        "candidate_id": str(row.candidate_id), "model_id": str(row.model_id),
        "model": str(row.model), "decoder": str(row.decoder),
        "protocol": PRIMARY_PROTOCOL, "development_metrics": metrics(row),
        "grouped_protocol": GROUPED_PROTOCOL, "grouped_metrics": metrics(grouped),
        "meets_development_thresholds": bool(row.accuracy > MIN_ACCURACY and row.min_f1 >= MIN_CLASS_F1),
    }


def record_selection() -> dict:
    """Select from a completed train-only experiment, verifying its provenance."""
    summary_path = FINAL_EVALUATION_DIR / "summary.csv"
    experiment_path = FINAL_EVALUATION_DIR / "experiment.json"
    run_state_path = FINAL_EVALUATION_DIR / "run_state.json"
    if not all(path.is_file() for path in (summary_path, experiment_path, run_state_path)):
        raise FileNotFoundError("Run run_final_evaluation.py before selecting the final model.")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    state = json.loads(run_state_path.read_text(encoding="utf-8"))
    if experiment.get("state") != "complete" or state.get("state") != "complete":
        raise ValueError("The final evaluation is still running or incomplete.")
    if experiment.get("seed") != SEED or experiment.get("LR_C") != LR_C or experiment.get("n_folds") != N_FOLDS:
        raise ValueError("Current settings differ from the evaluated experiment.")
    sources = experiment.get("sources_sha256", {})
    if sources.get("train.csv") != sha256(TRAIN_FILE):
        raise ValueError("Development scores do not match the current training file.")
    for name in IMPLEMENTATION_FILES:
        if sources.get(name) != sha256(ROOT / name):
            raise ValueError(f"{name} changed since evaluation; rerun the final evaluation.")
    artifact_hashes = experiment.get("artifacts_sha256", {})
    if artifact_hashes.get("summary.csv") != sha256(summary_path):
        raise ValueError("The summary differs from the completed evaluation manifest.")
    for name, expected in artifact_hashes.items():
        path = (FINAL_EVALUATION_DIR / name).resolve()
        if not path.is_relative_to(FINAL_EVALUATION_DIR.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Evaluation artifact changed or missing: {name}")
    selected = choose_candidate(pd.read_csv(summary_path, sep=";"))
    selected.update({
        "schema_version": 2, "selection_rule": SELECTION_RULE,
        "evaluation_status": EVALUATION_STATUS, "train_sha256": sources["train.csv"],
        "experiment_directory": "outputs/avaliacao_final",
        "summary_sha256": sha256(summary_path), "experiment_sha256": sha256(experiment_path),
        "implementation_sha256": {name: sha256(ROOT / name) for name in IMPLEMENTATION_FILES},
        "inference_implementation_sha256": {name: sha256(ROOT / name) for name in ("classify.py", "read_test_set.py", "config.py")},
        "LR_C": LR_C, "seed": SEED, "n_folds": N_FOLDS,
    })
    SELECTED_MODEL_FILE.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected


def load_selection() -> dict:
    selected = json.loads(SELECTED_MODEL_FILE.read_text(encoding="utf-8"))
    if selected.get("schema_version") != 2 or selected.get("candidate_id") not in CANDIDATE_IDS:
        raise ValueError("Legacy or unsupported selection; run the final evaluation and --select.")
    expected_id = selected.get("model_id", "") + ("+EXCL" if selected.get("decoder") == "exclude_seen" else "")
    if selected.get("decoder") not in DECODERS or selected["candidate_id"] != expected_id:
        raise ValueError("Selection contains an inconsistent classifier/decoder.")
    if selected.get("train_sha256") != sha256(TRAIN_FILE):
        raise ValueError("Selection is incompatible with the current training file.")
    if selected.get("LR_C") != LR_C or selected.get("seed") != SEED or selected.get("n_folds") != N_FOLDS:
        raise ValueError("Settings changed; rerun the final evaluation and selection.")
    if selected.get("protocol") != PRIMARY_PROTOCOL or selected.get("grouped_protocol") != GROUPED_PROTOCOL:
        raise ValueError("Selection uses an unsupported evaluation protocol.")
    required_hashes = {"implementation_sha256": set(IMPLEMENTATION_FILES),
                       "inference_implementation_sha256": {"classify.py", "read_test_set.py", "config.py"}}
    for field, names in required_hashes.items():
        if set(selected.get(field, {})) != names:
            raise ValueError("Selection is missing implementation provenance; regenerate it.")
        for name, expected in selected[field].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"{name} changed since selection; regenerate selection as appropriate.")
    return selected


def predict_selected(fitted, exclusion_index, test: pd.DataFrame, selected: dict):
    """The same score/decoder helpers used in CV, with no inference targets."""
    scores, classes = score_model(fitted, test, selected["model_id"])
    result = exclusion_index.predict(scores, classes, test)
    predictions = result.predictions if selected["decoder"] == "exclude_seen" else result.raw_predictions
    return [str(value) for value in predictions], result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true", help="select from both scenarios using row development only")
    args = parser.parse_args(argv)
    selected = record_selection() if args.select or not SELECTED_MODEL_FILE.exists() else load_selection()
    records, _ = repair_training_csv(TRAIN_FILE)
    train = pd.DataFrame(records)
    with threadpool_limits(limits=1):
        fitted, batch = fit_evaluated_model(train, selected["model_id"], seed=SEED, fold=0)
    # Classifier and original-training index are fixed before official inputs.
    exclusion_index = fit_exclusion_index(train)
    test_cases = read_test_cases(TEST_FILE)
    predicted, result = predict_selected(fitted, exclusion_index, pd.DataFrame(test_cases), selected)
    write_results(RESULTS_FILE, predicted, test_cases)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics = result.diagnostics.copy()
    diagnostics.insert(0, "case_id", [case["case_id"] for case in test_cases])
    diagnostics["raw_prediction"] = result.raw_predictions
    diagnostics["predicted_class"] = predicted
    diagnostics_path = FINAL_OUTPUT_DIR / "decoding_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, sep=";", encoding="utf-8-sig", index=False)
    final_metadata = {
        "selected_model": selected, "original_training_cases": len(train),
        "fit_rows_including_descendants": len(batch.y), "training_batch": batch.metadata,
        "decoder": selected["decoder"], "exclusion_index_original_training_rows": exclusion_index.original_training_rows,
        "exclusion_key": "lowercase whitespace-normalized (sample_name, transcription); skip only wholly empty input",
        "exclusion_fallback": "keep raw prediction for no match or all classes excluded",
        "test_source_sha256": sha256(TEST_FILE), "test_cases": len(test_cases),
        "results_sha256": sha256(RESULTS_FILE), "results_lines": len(predicted),
        "diagnostics_sha256": sha256(diagnostics_path),
        "test_metrics": None, "test_metrics_reason": "No official test labels are available.",
        "versions": {name: metadata.version(name) for name in ("pandas", "numpy", "scikit-learn")},
    }
    (FINAL_OUTPUT_DIR / "inference.json").write_text(
        json.dumps(final_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Selected {selected['candidate_id']}: {selected['model']} / {selected['decoder']}")
    print(f"Fit on all {len(train)} original cases; wrote {len(predicted)} predictions to {RESULTS_FILE}.")
    print("Row/grouped scores are exploratory development CV. Official test scores are unknown.")


if __name__ == "__main__":
    main()
