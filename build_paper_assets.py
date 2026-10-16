"""Build paper assets from the frozen two-protocol development evaluation.

This module does not select or fit a model, inspect the official test, or edit
paper.tex. Call ``main([])`` from a notebook, or run this file from the CLI.
Interpretation of the generated error candidates remains a manual paper task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / "outputs" / "avaliacao_recombinacao"
DEFAULT_SELECTION = ROOT / "selected_model.json"
DEFAULT_OUTPUT = ROOT / "paper_assets"
ORDER = ("NB", "B", "D", "E", "SVC", "SVC_AUG", "SVC_COPY")
ROW_PROTOCOL = "stratified_rows_development"
GROUP_PROTOCOL = "grouped_development"
PROTOCOLS = (ROW_PROTOCOL, GROUP_PROTOCOL)
DECODERS = ("raw", "exclude_seen")
# Editorial examples requested after inspecting the primary OOF errors. They
# are used only while they remain errors of the explicitly selected candidate.
PREFERRED_ERROR_IDS = ("1892", "309", "1383")
SYSTEM_NAMES = {
    "NB": "NB: description", "B": "LR: balanced originals",
    "D": "LR: balanced + section recombination", "E": "LR: balanced + copies",
    "SVC": "SVC: balanced originals", "SVC_AUG": "SVC: balanced + section recombination",
    "SVC_COPY": "SVC: balanced + copies",
}
TABLE_NAMES = {"NB": "NB", "B": "LR", "D": "LR+mix", "E": "LR+copy",
               "SVC": "SVC", "SVC_AUG": "SVC+mix", "SVC_COPY": "SVC+copy"}
CONFUSION_LABELS = {
    "Cardiovascular-Pulmonary": "Cardio-Pulm.", "Dermatology": "Dermatol.",
    "Gastroenterology": "Gastroent.", "General Medicine": "Gen. Med.",
    "Neurology": "Neurology", "Neurosurgery": "Neurosurg.",
    "Obstetrics-Gynecology": "Ob-Gyn.", "Ophthalmology": "Ophthalm.",
    "Orthopedic": "Orthopedic", "Psychiatry-Psychology": "Psych.-Psych.",
    "Radiology": "Radiology", "Surgery": "Surgery",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not 0 <= value <= 1:
        raise ValueError(f"{key} is not a metric in [0, 1]: {value}")
    return value


def _candidate(model_id, decoder):
    return model_id if decoder == "raw" else model_id + "+EXCL"


def _indexed(rows):
    indexed = {}
    for row in rows:
        key = (row["protocol"], row["candidate_id"])
        if key in indexed:
            raise ValueError(f"Repeated summary candidate: {key}")
        if row["model_id"] not in ORDER or row["decoder"] not in DECODERS:
            raise ValueError(f"Unsupported model or decoder: {key}")
        if row["candidate_id"] != _candidate(row["model_id"], row["decoder"]):
            raise ValueError(f"Candidate identity disagrees with model/decoder: {key}")
        if int(row["seed"]) != 0:
            raise ValueError("Paper assets require the frozen seed-0 comparison.")
        indexed[key] = row
    expected = {(protocol, _candidate(model, decoder))
                for protocol in PROTOCOLS for model in ORDER for decoder in DECODERS}
    if set(indexed) != expected:
        raise ValueError("Expected all 28 model/decoder/protocol summary rows.")
    return indexed


def _read_confusion(path: Path, classes: list[str]):
    import numpy as np

    rows = _rows(path)
    if len(rows) != len(classes) or not rows:
        raise ValueError("Confusion matrix has the wrong number of classes.")
    index_column = next(iter(rows[0]))
    if set(rows[0]) - {index_column} != set(classes):
        raise ValueError("Confusion-matrix columns do not match per-class labels.")
    indexed = {row[index_column]: row for row in rows}
    if set(indexed) != set(classes):
        raise ValueError("Confusion-matrix rows do not match per-class labels.")
    matrix = np.array([[float(indexed[actual][pred]) for pred in classes]
                       for actual in classes])
    if not np.isfinite(matrix).all() or (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError("Expected a finite nonnegative count confusion matrix.")
    return matrix.astype(int)


def _plot_confusion(matrix, output: Path, classes: list[str]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if matrix.shape != (len(classes), len(classes)) or any(label not in CONFUSION_LABELS for label in classes):
        raise ValueError("Confusion display labels must match the matrix classes.")
    figure, axis = plt.subplots(figsize=(3.35, 3.45), dpi=300)
    axis.imshow(matrix, cmap="Blues", interpolation="nearest", vmin=0)
    ticks = np.arange(len(matrix))
    labels = [CONFUSION_LABELS[label] for label in classes]
    axis.set_xticks(ticks, labels, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_yticks(ticks, labels)
    axis.set_xlabel("Predicted class", fontsize=8)
    axis.set_ylabel("Supplied class", fontsize=8)
    axis.tick_params(axis="both", labelsize=7, length=0, pad=2)
    threshold = matrix.max() * 0.65
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center",
                      fontsize=6.7, color="white" if matrix[row, col] > threshold else "#17212b")
    axis.set_xticks(np.arange(-0.5, len(matrix), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.25)
    axis.tick_params(which="minor", bottom=False, left=False)
    figure.tight_layout(pad=0.35)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _close(actual, expected, context):
    if abs(float(actual) - float(expected)) > 1e-10:
        raise ValueError(f"Inconsistent {context}: {actual} versus {expected}")


def _matrix_metrics(matrix, classes):
    import numpy as np

    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(len(classes)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support != 0)
    f1 = np.divide(2 * tp, predicted + support, out=np.zeros(len(classes)), where=(predicted + support) != 0)
    return {
        "accuracy": float(tp.sum() / support.sum()), "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()), "f1_macro": float(f1.mean()),
        "f1_weighted": float(np.dot(f1, support) / support.sum()), "min_f1": float(f1.min()),
        "zero_f1_classes": int((f1 == 0).sum()),
    }, [{"class": label, "support": int(support[i]), "precision": float(precision[i]),
         "recall": float(recall[i]), "f1": float(f1[i])} for i, label in enumerate(classes)]


def _validate_evaluation(experiment, summaries, records):
    """Recompute every count and metric from OOF rows, then check all 28 CMs."""
    import numpy as np

    originals = {str(r["record_id"]): r for r in records}
    classes = sorted({r["medical_specialty"] for r in records})
    if len(classes) != 12:
        raise ValueError("Expected exactly twelve training classes.")
    positions = {label: i for i, label in enumerate(classes)}
    oof = {key: {} for key in summaries}
    per_class_rows = {key: {} for key in summaries}
    for row in _rows(experiment / "metrics_per_class.csv"):
        key = (row["protocol"], row["candidate_id"])
        if key not in summaries or row["class"] in per_class_rows[key]:
            raise ValueError("Unknown or repeated per-class metric row.")
        if row["model_id"] != summaries[key]["model_id"] or row["decoder"] != summaries[key]["decoder"] or int(row["seed"]) != 0:
            raise ValueError("Per-class identity disagrees with summary.")
        per_class_rows[key][row["class"]] = row
    assignments = {protocol: {} for protocol in PROTOCOLS}
    for row in _rows(experiment / "oof_predictions.csv"):
        key, rid = (row["protocol"], row["candidate_id"]), row["record_id"]
        if key not in summaries or rid not in originals or rid in oof[key]:
            raise ValueError("Unknown or repeated OOF record/candidate.")
        if row["model_id"] != summaries[key]["model_id"] or row["decoder"] != summaries[key]["decoder"] or int(row["seed"]) != 0:
            raise ValueError("OOF identity disagrees with summary.")
        if row["true_class"] != originals[rid]["medical_specialty"] or row["predicted_class"] not in positions:
            raise ValueError("OOF classes disagree with the supplied training records.")
        if not 1 <= int(row["fold"]) <= 5:
            raise ValueError("Expected five-fold OOF assignments.")
        assignment = (row["fold"], row["group_id"])
        previous = assignments[row["protocol"]].setdefault(rid, assignment)
        if previous != assignment:
            raise ValueError("Candidates use different folds or group IDs.")
        oof[key][rid] = row
    for key in summaries:
        if set(oof[key]) != set(originals) or set(per_class_rows[key]) != set(classes):
            raise ValueError(f"Incomplete OOF predictions or per-class metrics: {key}")
    grouped_notes, grouped_ids = {}, {}
    for rid, source in originals.items():
        fold, group_id = assignments[GROUP_PROTOCOL][rid]
        transcription, title = _normalise(source["transcription"]), _normalise(source["sample_name"])
        note = ("transcription", transcription) if transcription else ("title", title) if title else ("missing", rid)
        grouped_notes.setdefault(note, set()).add(fold)
        grouped_ids.setdefault(group_id, set()).add(fold)
    if any(len(folds) > 1 for folds in [*grouped_notes.values(), *grouped_ids.values()]):
        raise ValueError("Grouped note identity crosses validation folds.")
    matrices, computed_classes, paths = {}, {}, []
    for key, summary in summaries.items():
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for row in oof[key].values():
            matrix[positions[row["true_class"]], positions[row["predicted_class"]]] += 1
            raw = oof[(key[0], summary["model_id"])][row["record_id"]]
            if row["raw_prediction"] != raw["predicted_class"]:
                raise ValueError("OOF raw_prediction does not agree with its raw candidate.")
        path = experiment / key[0] / f"confusion_{key[1]}.csv"
        if not np.array_equal(matrix, _read_confusion(path, classes)):
            raise ValueError(f"Saved confusion matrix does not match OOF predictions: {key}")
        metrics, class_metrics = _matrix_metrics(matrix, classes)
        for name, value in metrics.items():
            if name in summary:
                _close(value, summary[name], f"OOF/summary {key} {name}")
        for row in class_metrics:
            saved = per_class_rows[key][row["class"]]
            for name in ("support", "precision", "recall", "f1"):
                _close(row[name], saved[name], f"OOF/per-class {key} {row['class']} {name}")
        if "f1_dermatology" in summary:
            value = next(row["f1"] for row in class_metrics if row["class"] == "Dermatology")
            _close(value, summary["f1_dermatology"], f"OOF/summary {key} Dermatology F1")
        worst = summary["worst_class"]
        if worst not in classes:
            raise ValueError("Unknown worst class in summary.")
        _close(next(r["f1"] for r in class_metrics if r["class"] == worst), metrics["min_f1"], "worst-class F1")
        matrices[key], computed_classes[key] = matrix, class_metrics
        paths.append(path)
    for model in ORDER:
        raw = oof[(GROUP_PROTOCOL, model)]
        masked = oof[(GROUP_PROTOCOL, model + "+EXCL")]
        if any(raw[rid]["predicted_class"] != masked[rid]["predicted_class"] for rid in originals):
            raise ValueError("Grouped exclusion is not a no-op; expand the global table before omitting it.")
    return classes, oof, matrices, computed_classes, paths


def _error_candidates(records, oof, selection):
    """Select three remaining errors; retain one correction as optional context."""
    model, chosen_id = selection["model_id"], selection["candidate_id"]
    raw, masked = oof[(ROW_PROTOCOL, model)], oof[(ROW_PROTOCOL, model + "+EXCL")]
    peers = {}
    for record in records:
        key = tuple(_normalise(record[field]) for field in ("sample_name", "transcription"))
        if any(key):
            peers.setdefault(key, []).append(record)
    candidates = []
    for source in records:
        rid = str(source["record_id"])
        plain, decoded = raw[rid], masked[rid]
        gold = source["medical_specialty"]
        if plain["predicted_class"] == gold and decoded["predicted_class"] == gold:
            continue
        key = tuple(_normalise(source[field]) for field in ("sample_name", "transcription"))
        training_peers = [r for r in peers.get(key, []) if raw[str(r["record_id"])]["fold"] != plain["fold"]]
        seen = sorted({r["medical_specialty"] for r in training_peers})
        if "seen_classes" in decoded and sorted(json.loads(decoded["seen_classes"])) != seen:
            raise ValueError("Saved decoder diagnostics disagree with original training-fold matches.")
        corrected = plain["predicted_class"] != gold and decoded["predicted_class"] == gold
        category = ("raw_error_corrected" if corrected else "true_label_blocked" if gold in seen else
                    "matched_error_persists" if seen else "unmatched_error")
        excerpt = " ".join(re.sub(r"\s+", " ", source["transcription"] or source["description"]).strip().split()[:55])
        selected_prediction = oof[(ROW_PROTOCOL, chosen_id)][rid]["predicted_class"]
        candidates.append({
            "category": category, "protocol": ROW_PROTOCOL, "record_id": rid, "fold": plain["fold"],
            "selected_candidate_id": chosen_id, "sample_name": source["sample_name"].strip(),
            "excerpt": excerpt, "true_class": gold, "raw_prediction": plain["predicted_class"],
            "decoded_prediction": decoded["predicted_class"],
            "selected_prediction": selected_prediction, "selected_error": selected_prediction != gold,
            "has_train_match": bool(seen), "mask_applied": bool(seen) and len(seen) < len({r["medical_specialty"] for r in records}),
            "true_label_in_excluded_classes": gold in seen,
            "title_only_match": bool(seen) and not source["transcription"].strip(),
            "n_excluded_classes": len(seen), "raw_prediction_excluded": plain["predicted_class"] in seen,
            "decoded_choice_raw_rank": decoded.get("decoded_choice_raw_rank", ""),
            "fallback_all_excluded": decoded.get("fallback_all_excluded", ""),
            "excluded_classes": json.dumps(seen, ensure_ascii=False),
            "training_peer_record_ids": json.dumps([r["record_id"] for r in training_peers]),
            "corpus_annotation_labels": json.dumps(sorted({r["medical_specialty"] for r in peers.get(key, [])})),
            "transcription_characters": len(source["transcription"]),
            "interpretation_basis": (
                "Only training-fold labels of identical available normalized (sample_name, transcription) inputs form the exclusion set. "
                "These examples describe decoder behavior; they do not establish the clinical cause of the raw error."
            ),
        })
    remaining = [r for r in candidates if r["selected_error"]]
    by_id = {r["record_id"]: r for r in remaining}
    chosen = [by_id[rid] for rid in PREFERRED_ERROR_IDS if rid in by_id]
    for allowed in (("true_label_blocked", "matched_error_persists"), ("unmatched_error",)):
        if len(chosen) >= 3:
            break
        found = next((r for category in allowed for r in remaining if r["category"] == category and r not in chosen), None)
        if found:
            chosen.append(found)
    # Prefer a distinct supplied class for the third example, without inferring
    # feature importance or clinical causes from its vocabulary.
    ordered = sorted(remaining, key=lambda r: (r["true_class"] in {c["true_class"] for c in chosen}, int(r["record_id"])))
    for row in ordered:
        if len(chosen) >= 3:
            break
        if row not in chosen:
            chosen.append(row)
    if len(chosen) < 3:
        raise ValueError("Fewer than three distinct selected-candidate OOF errors are available.")
    correction = next((r for r in candidates if r["category"] == "raw_error_corrected"), None)
    return chosen, correction


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _tabular(path, alignment, header, rows):
    path.write_text("\n".join([
        r"\begin{tabular}{@{}" + alignment + r"@{}}", r"\toprule", header + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows], r"\bottomrule", r"\end{tabular}",
    ]) + "\n", encoding="utf-8")


def build_assets(experiment: Path, selection_path: Path, output: Path, *, train_path: Path | None = None) -> dict:
    """Validate and render the explicitly selected row candidate; never select or fit."""
    from data_preparation import repair_training_csv

    train_path = ROOT / "train.csv" if train_path is None else train_path
    if not selection_path.is_file():
        return {"status": "waiting_for_selection", "selection_file": str(selection_path)}
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    if selection.get("schema_version") != 3:
        return {"status": "waiting_for_schema_3_selection", "selection_file": str(selection_path)}
    required_files = [experiment / name for name in (
        "summary.csv", "experiment.json", "metrics_per_class.csv", "oof_predictions.csv", "gate_diagnostics.csv",
    )]
    if any(not path.is_file() for path in required_files):
        return {"status": "waiting_for_experiment", "missing_files": [str(p) for p in required_files if not p.is_file()]}
    design = json.loads((experiment / "experiment.json").read_text(encoding="utf-8-sig"))
    if design.get("state") != "complete":
        raise ValueError("The final evaluation is not complete.")
    state_path = experiment / "run_state.json"
    if state_path.is_file() and json.loads(state_path.read_text(encoding="utf-8-sig")).get("state") != "complete":
        raise ValueError("A new evaluation run is still in progress.")
    for key, path in (("train_sha256", train_path), ("summary_sha256", experiment / "summary.csv"),
                      ("experiment_sha256", experiment / "experiment.json")):
        if selection.get(key) != _sha256(path):
            raise ValueError(f"Selection/source hash mismatch: {key}")
    source_hash = design.get("source_sha256") or design.get("sources_sha256", {}).get("train.csv")
    if source_hash != _sha256(train_path):
        raise ValueError("Evaluation metadata belongs to a different training corpus.")
    manifest = design.get("artifacts_sha256")
    if not isinstance(manifest, dict):
        raise ValueError("Missing evaluation artifact-hash manifest.")
    used_artifacts = [p for p in required_files if p.name != "experiment.json"]
    used_artifacts += [experiment / protocol / f"confusion_{_candidate(model_id, decoder)}.csv"
                      for protocol in PROTOCOLS for model_id in ORDER for decoder in DECODERS]
    for path in used_artifacts:
        name = path.relative_to(experiment).as_posix()
        if manifest.get(name) != _sha256(path):
            raise ValueError(f"Evaluation artifact hash mismatch: {name}")
    selected_id, model = selection["candidate_id"], selection["model_id"]
    if selection["protocol"] != ROW_PROTOCOL or model not in ORDER or selection["decoder"] not in DECODERS:
        raise ValueError("Paper requires an explicitly selected row-protocol candidate.")
    if selected_id != _candidate(model, selection["decoder"]):
        raise ValueError("Selected candidate identity disagrees with model/decoder.")
    summaries = _indexed(_rows(experiment / "summary.csv"))
    selected = summaries[(ROW_PROTOCOL, selected_id)]
    metric_names = ("accuracy", "f1_macro", "min_f1")
    if "f1_dermatology" in selected:
        metric_names += ("f1_dermatology",)
    for name in metric_names:
        if name not in selection.get("development_metrics", {}):
            raise ValueError(f"Missing selected metric: {name}")
        _close(selection["development_metrics"][name], _number(selected, name), f"selection {name}")
    records, _ = repair_training_csv(train_path)
    classes, oof, matrices, per_class, matrix_paths = _validate_evaluation(experiment, summaries, records)
    # The runner keeps per-class values in metrics_per_class.csv; derive this
    # gate metric from the already checked count matrices even if summary.csv
    # does not repeat it as a separate column.
    for key, summary in summaries.items():
        summary["f1_dermatology"] = next(row["f1"] for row in per_class[key] if row["class"] == "Dermatology")
    if "f1_dermatology" not in selection.get("development_metrics", {}):
        raise ValueError("Missing selected metric: f1_dermatology")
    _close(selection["development_metrics"]["f1_dermatology"], selected["f1_dermatology"],
           "selection Dermatology F1")
    if "f1_dermatology" not in metric_names:
        metric_names += ("f1_dermatology",)
    examples, correction_example = _error_candidates(records, oof, selection)
    gate_rows = _rows(experiment / "gate_diagnostics.csv")
    if not gate_rows:
        raise ValueError("The recorded promotion diagnostics are empty.")
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the frozen decision evidence without selecting a candidate here.
    (output / "gate_diagnostics.csv").write_bytes((experiment / "gate_diagnostics.csv").read_bytes())
    table_rows, global_rows = [], []
    for base in ("NB", "B", "SVC"):
        for protocol, decoder, scenario in ((ROW_PROTOCOL, "raw", "R"), (ROW_PROTOCOL, "exclude_seen", r"R+EX"), (GROUP_PROTOCOL, "raw", "G")):
            row = summaries[(protocol, _candidate(base, decoder))]
            cells = [TABLE_NAMES[base], scenario]
            cells += [f"{100 * _number(row, key):.2f}" for key in ("accuracy", "f1_macro", "min_f1")]
            if (protocol, row["candidate_id"]) == (ROW_PROTOCOL, selected_id):
                cells = [r"\textbf{" + cell + "}" for cell in cells]
            table_rows.append(cells)
            global_rows.append(row)
    extra_selected = model not in {"NB", "B", "SVC"}
    if extra_selected:
        for protocol, scenario in ((ROW_PROTOCOL, "R+EX" if selection["decoder"] == "exclude_seen" else "R"),
                                   (GROUP_PROTOCOL, "G")):
            row = summaries[(protocol, selected_id)]
            cells = [TABLE_NAMES[model], scenario]
            cells += [f"{100 * _number(row, key):.2f}" for key in ("accuracy", "f1_macro", "min_f1")]
            table_rows.append([r"\textbf{" + cell + "}" for cell in cells] if protocol == ROW_PROTOCOL else cells)
            global_rows.append(row)
    _tabular(output / "results.tex", "llrrr", r"System & CV & Acc. & Macro F1 & Min. F1", table_rows)
    short = {"Cardiovascular-Pulmonary": "Cardiovasc.-Pulm.", "Obstetrics-Gynecology": "Obstetrics-Gynecol.",
             "Psychiatry-Psychology": "Psychiatry-Psychol."}
    row_selected = per_class[(ROW_PROTOCOL, selected_id)]
    group_selected = per_class[(GROUP_PROTOCOL, selected_id)]
    row_raw, row_masked = per_class[(ROW_PROTOCOL, model)], per_class[(ROW_PROTOCOL, model + "+EXCL")]
    class_table, alternative, class_export = [], [], []
    for i, row in enumerate(row_selected):
        label = row["class"]
        prefix = [_tex(short.get(label, label)), str(row["support"])]
        class_table.append(prefix + [f"{100 * row[key]:.1f}" for key in ("precision", "recall", "f1")] + [f"{100 * group_selected[i]['f1']:.1f}"])
        alternative.append(prefix + [f"{100 * values[i]['f1']:.1f}" for values in (row_raw, row_masked, group_selected)])
        class_export.append({"class": label, "support": row["support"], "precision_row_selected": row["precision"],
                             "recall_row_selected": row["recall"], "f1_row_selected": row["f1"],
                             "f1_row_raw": row_raw[i]["f1"], "f1_row_excluded": row_masked[i]["f1"],
                             "f1_grouped_selected": group_selected[i]["f1"]})
    _tabular(output / "per_class.tex", "lrrrrr", r"Specialty & $n$ & $P_R$ & $R_R$ & $F1_R$ & $F1_G$", class_table)
    _tabular(output / "per_class_comparison.tex", "lrrrr", r"Specialty & $n$ & $F1_R$ & $F1_{R+EX}$ & $F1_G$", alternative)
    _write_csv(output / "per_class_comparison.csv", class_export)
    _plot_confusion(matrices[(ROW_PROTOCOL, selected_id)], output / "confusion_matrix.png", classes)
    _write_csv(output / "error_candidates.csv", examples)
    _write_csv(output / "global_comparison.csv", global_rows)
    augmentation_rows = [summaries[(protocol, _candidate(base, decoder))]
                         for protocol in PROTOCOLS for decoder in DECODERS
                         for base in ("B", "D", "E", "SVC", "SVC_AUG", "SVC_COPY")]
    _write_csv(output / "augmentation_summary.csv", augmentation_rows)
    deltas = []
    for protocol in PROTOCOLS:
        for decoder in DECODERS:
            for left, right, meaning in (("D", "B", "LR recombination versus originals"),
                                         ("D", "E", "LR recombination versus matched copies"),
                                         ("E", "B", "LR matched copies versus originals"),
                                         ("SVC_AUG", "SVC", "SVC recombination versus originals"),
                                         ("SVC_AUG", "SVC_COPY", "SVC recombination versus matched copies"),
                                         ("SVC_COPY", "SVC", "SVC matched copies versus originals")):
                first, second = (summaries[(protocol, _candidate(base, decoder))] for base in (left, right))
                deltas.append({"protocol": protocol, "decoder": decoder, "comparison": f"{left}-{right}", "meaning": meaning,
                               **{key + "_delta_pp": 100 * (_number(first, key) - _number(second, key))
                                  for key in metric_names}})
    metrics = {key: _number(selected, key) for key in metric_names}
    selected_matrix = matrices[(ROW_PROTOCOL, selected_id)]
    most_common_error = max(
        ({"true_class": actual, "predicted_class": predicted, "count": int(selected_matrix[i, j])}
         for i, actual in enumerate(classes) for j, predicted in enumerate(classes) if i != j),
        key=lambda entry: entry["count"],
    )
    paths = [selection_path, train_path, *required_files, *matrix_paths]
    artifact_metadata = {
        "schema_version": 3, "status": "exploratory development CV; official test never read",
        "selected_candidate_id": selected_id, "selected_model_id": model, "decoder": selection["decoder"],
        "model": selected["model"], "protocol": ROW_PROTOCOL, "metrics": metrics,
        "grouped_metrics_same_candidate": {key: _number(summaries[(GROUP_PROTOCOL, selected_id)], key) for key in metrics},
        "raw_row_metrics_same_model": {key: _number(summaries[(ROW_PROTOCOL, model)], key) for key in metrics},
        "excluded_row_metrics_same_model": {key: _number(summaries[(ROW_PROTOCOL, model + "+EXCL")], key) for key in metrics},
        "worst_class": selected["worst_class"], "class_order": classes, "per_class": class_export,
        "support": len(records), "grouped_exclusion_verified_noop_all_models": True,
        "validation": "All 28 OOF sets checked against original training IDs/labels, per-class metrics, summary and saved count matrices.",
        "augmentation_comparisons": deltas, "error_candidates": examples,
        "error_candidate_selection": "Editorial examples 1892/309/1383 when they remain selected-candidate errors; otherwise deterministic matched/unmatched fallbacks.",
        "optional_raw_error_correction": correction_example,
        "global_table_extra_selected_candidate": extra_selected,
        "table_scenarios": {"R": "row-stratified raw", "R+EX": "row-stratified full-input exclusion", "G": "note-grouped; exclusion verified no-op"},
        "confusion_matrix_scenario": {"protocol": ROW_PROTOCOL, "candidate_id": selected_id},
        "confusion_display_labels": {label: CONFUSION_LABELS[label] for label in classes},
        "selection_policy": selection.get("selection_policy", selection.get("selection_rule")),
        "promotion_diagnostics": selection.get("promotion_diagnostics"),
        "gate_diagnostics": gate_rows,
        "most_frequent_confusion": most_common_error,
        "body_pagination": "See output/pdf/compilation.json; its source hashes must match current paper/assets.",
        "source_sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): _sha256(p) for p in paths},
    }
    (output / "selected_metrics.json").write_text(json.dumps(artifact_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "ready", "selected_candidate_id": selected_id, "selected_model_id": model, "metrics": metrics,
            "output_dir": str(output), "error_candidates": str(output / "error_candidates.csv")}


def main(argv=None):
    """Notebook API: ``main([])`` or ``main()``; neither captures Jupyter arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args([] if argv is None else argv)
    result = build_assets(args.experiment_dir.resolve(), args.selection.resolve(), args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
