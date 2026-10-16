"""Selection must use the declared scenario, preserve class coverage and provenance."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import classify
from classify import choose_candidate
from data_preparation import sha256


class SelectionTests(unittest.TestCase):
    def summary(self):
        return pd.DataFrame([
            {"candidate_id": name + ("+EXCL" if decoder == "exclude_seen" else ""),
             "model_id": name, "decoder": decoder, "model": classify.MODEL_NAMES[name],
             "protocol": protocol, "seed": classify.SEED,
             "accuracy": .40, "f1_macro": .40, "min_f1": .20}
            for protocol in (classify.PRIMARY_PROTOCOL, classify.GROUPED_PROTOCOL)
            for name in sorted(classify.MODEL_IDS)
            for decoder in ("raw", "exclude_seen")
        ])

    def set_metrics(self, frame, candidate, metrics, protocol=None):
        mask = frame.candidate_id.eq(candidate) & frame.protocol.eq(protocol or classify.PRIMARY_PROTOCOL)
        frame.loc[mask, list(metrics)] = list(metrics.values())

    def test_class_constraint_precedes_accuracy(self):
        frame = self.summary()
        self.set_metrics(frame, "A", {"accuracy": .80, "min_f1": .10})
        self.set_metrics(frame, "B+EXCL", {"accuracy": .52, "min_f1": .30})
        chosen = choose_candidate(frame)
        self.assertEqual(chosen["candidate_id"], "B+EXCL")
        self.assertEqual(chosen["decoder"], "exclude_seen")

    def test_grouped_score_cannot_change_primary_selection(self):
        frame = self.summary()
        self.set_metrics(frame, "B", {"accuracy": .52, "min_f1": .30})
        self.set_metrics(frame, "SVC", {"accuracy": .99, "min_f1": .90}, classify.GROUPED_PROTOCOL)
        chosen = choose_candidate(frame)
        self.assertEqual(chosen["candidate_id"], "B")
        self.assertEqual(chosen["grouped_metrics"]["accuracy"], .40)

    def test_fallback_uses_macro_f1(self):
        frame = self.summary()
        self.set_metrics(frame, "D", {"accuracy": .35, "f1_macro": .60})
        self.assertEqual(choose_candidate(frame)["candidate_id"], "D")

    def test_accuracy_must_be_strictly_above_threshold(self):
        frame = self.summary()
        self.set_metrics(frame, "B", {"accuracy": .49, "min_f1": .25})
        self.assertFalse(choose_candidate(frame)["meets_development_thresholds"])

    def test_undeclared_protocol_or_seed_is_rejected(self):
        for column, value in (("protocol", "some_other_split"), ("seed", 99)):
            frame = self.summary()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                choose_candidate(frame)

    def test_partial_duplicate_inconsistent_or_nonfinite_experiments_are_rejected(self):
        base = self.summary()
        cases = [base.iloc[:-1], pd.concat([base, base.iloc[:1]])]
        for column, value in (("accuracy", float("nan")), ("accuracy", 1.1),
                              ("decoder", "unknown"), ("model_id", "SVC")):
            frame = base.copy()
            frame.loc[0, column] = value
            cases.append(frame)
        for frame in cases:
            with self.subTest(rows=len(frame)):
                with self.assertRaises(ValueError):
                    choose_candidate(frame)

    def test_tie_is_deterministic(self):
        frame = self.summary()
        self.assertEqual(choose_candidate(frame.sample(frac=1, random_state=1))["candidate_id"], "A")

    def fixture(self, directory):
        root = Path(directory)
        training = root / "train.csv"
        training.write_text("fixture", encoding="utf-8")
        names = classify.IMPLEMENTATION_FILES + ("classify.py", "read_test_set.py", "config.py")
        for name in names:
            (root / name).write_text("fixture code", encoding="utf-8")
        summary = root / "summary.csv"
        self.summary().to_csv(summary, sep=";", index=False)
        experiment = {
            "state": "complete", "seed": classify.SEED, "LR_C": classify.LR_C,
            "n_folds": classify.N_FOLDS,
            "sources_sha256": {"train.csv": sha256(training),
                              **{name: sha256(root / name) for name in classify.IMPLEMENTATION_FILES}},
            "artifacts_sha256": {"summary.csv": sha256(summary)},
        }
        (root / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
        (root / "run_state.json").write_text('{"state": "complete"}', encoding="utf-8")
        return root, training

    def test_selection_rejects_changed_implementation_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root, training = self.fixture(directory)
            (root / "run_state.json").write_text('{"state": "running"}', encoding="utf-8")
            with patch.multiple(classify, FINAL_EVALUATION_DIR=root, ROOT=root,
                                TRAIN_FILE=training, SELECTED_MODEL_FILE=root / "selected.json"):
                with self.assertRaisesRegex(ValueError, "running or incomplete"):
                    classify.record_selection()
                (root / "run_state.json").write_text('{"state": "complete"}', encoding="utf-8")
                (root / "duplicate_exclusion.py").write_text("changed after evaluation", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed since evaluation"):
                    classify.record_selection()
                self.assertFalse((root / "selected.json").exists())

    def test_manifest_detects_modified_summary_and_valid_selection_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root, training = self.fixture(directory)
            with patch.multiple(classify, FINAL_EVALUATION_DIR=root, ROOT=root,
                                TRAIN_FILE=training, SELECTED_MODEL_FILE=root / "selected.json"):
                selected = classify.record_selection()
                self.assertEqual(selected, classify.load_selection())
                with (root / "summary.csv").open("a", encoding="utf-8") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, "manifest"):
                    classify.record_selection()

    def test_legacy_selection_requires_explicit_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            path.write_text('{"schema_version": 1, "model_id": "SVC"}', encoding="utf-8")
            with patch.object(classify, "SELECTED_MODEL_FILE", path):
                with self.assertRaisesRegex(ValueError, "Legacy"):
                    classify.load_selection()


if __name__ == "__main__":
    unittest.main()
