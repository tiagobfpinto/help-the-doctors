"""Asset provenance checks with synthetic OOF fixtures; no fitting or test data."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import build_paper_assets as assets
from data_preparation import COLUMNS, SPECIALTIES


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    train = root / "train.csv"
    labels = sorted(SPECIALTIES)
    records = []
    for i, label in enumerate(labels):
        for j in range(4):
            records.append({
                "medical_specialty": label, "description": f"Description {i}-{j}",
                "sample_name": f"Shared {i // 2}" if j == 0 else f"Name {i}-{j}",
                "transcription": f"The shared note {i // 2}" if j == 0 else f"The unique note {i}-{j}",
                "keywords": "",
            })
    write_csv(train, [{field: row[field] for field in COLUMNS} for row in records])
    experiment = root / "evaluation"
    summaries, class_rows, oof = [], [], []
    for protocol in assets.PROTOCOLS:
        for model in assets.ORDER:
            for decoder in assets.DECODERS:
                candidate = assets._candidate(model, decoder)
                matrix = np.zeros((12, 12), dtype=int)
                for index, row in enumerate(records):
                    i, j = divmod(index, 4)
                    raw = labels[i ^ 1] if j == 0 else labels[(i + 1) % 12] if j == 1 else labels[i]
                    pred = raw
                    if protocol == assets.ROW_PROTOCOL and decoder == "exclude_seen" and j == 0:
                        pred = labels[i] if i % 2 == 0 else labels[(i + 2) % 12]
                    fold = (i % 2) + 1 if j == 0 else (i + j) % 5 + 1
                    if protocol == assets.GROUP_PROTOCOL and j == 0:
                        fold = (i // 2) % 5 + 1
                    oof.append({
                        "protocol": protocol, "seed": 0, "candidate_id": candidate,
                        "model_id": model, "decoder": decoder, "record_id": index + 1,
                        "true_class": row["medical_specialty"], "predicted_class": pred,
                        "raw_prediction": raw, "fold": fold, "group_id": f"single-{index}" if j else f"pair-{i // 2}",
                    })
                    matrix[i, labels.index(pred)] += 1
                metrics, per_class = assets._matrix_metrics(matrix, labels)
                metrics["f1_dermatology"] = next(r["f1"] for r in per_class if r["class"] == "Dermatology")
                summaries.append({
                    "protocol": protocol, "seed": 0, "candidate_id": candidate, "model_id": model,
                    "decoder": decoder, "model": model, **metrics,
                    "worst_class": min(per_class, key=lambda r: r["f1"])["class"],
                })
                for item in per_class:
                    class_rows.append({"protocol": protocol, "seed": 0, "candidate_id": candidate,
                                       "model_id": model, "decoder": decoder, **item})
                write_csv(experiment / protocol / f"confusion_{candidate}.csv", [
                    {"true_class": label, **{other: int(matrix[i, k]) for k, other in enumerate(labels)}}
                    for i, label in enumerate(labels)
                ])
    write_csv(experiment / "summary.csv", summaries)
    write_csv(experiment / "metrics_per_class.csv", class_rows)
    write_csv(experiment / "oof_predictions.csv", oof)
    write_csv(experiment / "gate_diagnostics.csv", [
        {"candidate_id": candidate, "eligible": False, "reason": "synthetic fixture"}
        for candidate in ("D+EXCL", "SVC_AUG+EXCL")
    ])
    manifest = {p.relative_to(experiment).as_posix(): assets._sha256(p) for p in experiment.rglob("*.csv")}
    (experiment / "experiment.json").write_text(json.dumps({"state": "complete", "source_sha256": assets._sha256(train),
                                                            "artifacts_sha256": manifest}), encoding="utf-8")
    chosen = next(r for r in summaries if r["candidate_id"] == "SVC+EXCL" and r["protocol"] == assets.ROW_PROTOCOL)
    selection = root / "selection.json"
    selection.write_text(json.dumps({
        "schema_version": 3, "candidate_id": "SVC+EXCL", "model_id": "SVC", "decoder": "exclude_seen",
        "protocol": assets.ROW_PROTOCOL,
        "development_metrics": {k: chosen[k] for k in ("accuracy", "f1_macro", "min_f1", "f1_dermatology")},
        "train_sha256": assets._sha256(train), "summary_sha256": assets._sha256(experiment / "summary.csv"),
        "experiment_sha256": assets._sha256(experiment / "experiment.json"),
    }), encoding="utf-8")
    return train, experiment, selection


class PaperAssetTests(unittest.TestCase):
    def test_complete_assets_validate_both_protocols_without_a_test_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            output = root / "assets"
            with patch.object(assets, "_plot_confusion", side_effect=lambda matrix, path, classes: path.write_bytes(b"plot fixture")):
                result = assets.build_assets(experiment, selection, output, train_path=train)
            self.assertEqual(result["selected_candidate_id"], "SVC+EXCL")
            metadata = json.loads((output / "selected_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["support"], 48)
            self.assertTrue(metadata["grouped_exclusion_verified_noop_all_models"])
            self.assertEqual(metadata["confusion_matrix_scenario"]["protocol"], assets.ROW_PROTOCOL)
            self.assertEqual(len(assets._rows(output / "global_comparison.csv")), 9)
            self.assertEqual(len(metadata["per_class"]), 12)
            self.assertEqual(len(metadata["error_candidates"]), 3)
            self.assertTrue(all(r["selected_prediction"] != r["true_class"] for r in metadata["error_candidates"]))
            self.assertEqual({r["category"] for r in metadata["error_candidates"]},
                             {"matched_error_persists", "unmatched_error"})
            self.assertEqual(metadata["optional_raw_error_correction"]["category"], "raw_error_corrected")
            self.assertIn("R+EX", (output / "results.tex").read_text(encoding="utf-8"))
            self.assertEqual((output / "gate_diagnostics.csv").read_bytes(),
                             (experiment / "gate_diagnostics.csv").read_bytes())
            self.assertFalse((root / "test_no_labels.csv").exists())

    def test_augmented_svc_selection_is_named_and_reported_as_svc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            chosen = json.loads(selection.read_text(encoding="utf-8"))
            chosen.update(candidate_id="SVC_AUG+EXCL", model_id="SVC_AUG")
            selection.write_text(json.dumps(chosen), encoding="utf-8")
            output = root / "assets"
            with patch.object(assets, "_plot_confusion", side_effect=lambda matrix, path, classes: path.write_bytes(b"plot fixture")):
                result = assets.build_assets(experiment, selection, output, train_path=train)
            self.assertEqual(result["selected_model_id"], "SVC_AUG")
            self.assertIn(r"\textbf{SVC+mix}", (output / "results.tex").read_text(encoding="utf-8"))
            self.assertEqual(len(assets._rows(output / "global_comparison.csv")), 11)

    def test_changed_promotion_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            path = experiment / "gate_diagnostics.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                assets.build_assets(experiment, selection, root / "assets", train_path=train)

    def test_stale_summary_hash_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            path = experiment / "summary.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                assets.build_assets(experiment, selection, root / "assets", train_path=train)

    def test_changed_oof_predictions_cannot_reuse_old_metrics_or_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            path = experiment / "oof_predictions.csv"
            rows = assets._rows(path)
            rows[0]["predicted_class"] = rows[0]["true_class"]
            rows[0]["raw_prediction"] = rows[0]["true_class"]
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                assets.build_assets(experiment, selection, root / "assets", train_path=train)

    def test_per_class_precision_is_checked_against_oof_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            path = experiment / "metrics_per_class.csv"
            rows = assets._rows(path)
            rows[0]["precision"] = ".123"
            write_csv(path, rows)
            # Refresh declared hashes to exercise count consistency separately
            # from byte-level provenance checks.
            metadata_path = experiment / "experiment.json"
            design = json.loads(metadata_path.read_text(encoding="utf-8"))
            design["artifacts_sha256"]["metrics_per_class.csv"] = assets._sha256(path)
            metadata_path.write_text(json.dumps(design), encoding="utf-8")
            chosen = json.loads(selection.read_text(encoding="utf-8"))
            chosen["experiment_sha256"] = assets._sha256(metadata_path)
            selection.write_text(json.dumps(chosen), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "OOF/per-class"):
                assets.build_assets(experiment, selection, root / "assets", train_path=train)

    def test_dermatology_gate_metric_is_recomputed_from_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            path = experiment / "summary.csv"
            rows = assets._rows(path)
            rows[0]["f1_dermatology"] = ".123"
            write_csv(path, rows)
            metadata_path = experiment / "experiment.json"
            design = json.loads(metadata_path.read_text(encoding="utf-8"))
            design["artifacts_sha256"]["summary.csv"] = assets._sha256(path)
            metadata_path.write_text(json.dumps(design), encoding="utf-8")
            chosen = json.loads(selection.read_text(encoding="utf-8"))
            chosen["summary_sha256"] = assets._sha256(path)
            chosen["experiment_sha256"] = assets._sha256(metadata_path)
            selection.write_text(json.dumps(chosen), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "OOF/summary.*Dermatology F1"):
                assets.build_assets(experiment, selection, root / "assets", train_path=train)

    def test_legacy_selection_is_not_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection.json"
            selection.write_text('{"schema_version":1,"model_id":"SVC"}', encoding="utf-8")
            result = assets.build_assets(root / "absent", selection, root / "assets")
            self.assertEqual(result["status"], "waiting_for_schema_3_selection")

    def test_grouped_note_split_is_checked_even_when_metrics_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, experiment, selection = fixture(root)
            path = experiment / "oof_predictions.csv"
            rows = assets._rows(path)
            for row in rows:
                if row["protocol"] == assets.GROUP_PROTOCOL and row["record_id"] == "1":
                    row["fold"] = "2"
            write_csv(path, rows)
            metadata_path = experiment / "experiment.json"
            design = json.loads(metadata_path.read_text(encoding="utf-8"))
            design["artifacts_sha256"]["oof_predictions.csv"] = assets._sha256(path)
            metadata_path.write_text(json.dumps(design), encoding="utf-8")
            chosen = json.loads(selection.read_text(encoding="utf-8"))
            chosen["experiment_sha256"] = assets._sha256(metadata_path)
            selection.write_text(json.dumps(chosen), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Grouped note identity"):
                assets.build_assets(experiment, selection, root / "assets", train_path=train)


if __name__ == "__main__":
    unittest.main()
