"""Selection verifies completed evidence; normal release inference is standalone."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import augmentation_policy as policy
import classify
from data_preparation import sha256
from tests import test_augmentation_policy as policy_tests


class SelectionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        output = root / "experiment"
        output.mkdir()
        for name in {*policy.SOURCE_FILES, *classify.RELEASE_FILES} - {policy.PLAN_FILENAME}:
            (root / name).write_text("fixture code or training data", encoding="utf-8")
        example = root / "review.json"
        example.write_text("[]", encoding="utf-8")
        plan = {
            "status": "frozen", "approved_to_fit": True, "policy_spec": policy.POLICY_SPEC,
            "review": {"reviewed_count": 5, "reviewer": "test reviewer", "examples_file": example.name,
                       "examples_sha256": sha256(example)},
            "settings": {"seed": 0, "n_folds": 5, "LR_C": .25, "model_ids": list(policy.MODEL_IDS), "protocols": list(policy.PROTOCOLS)},
            "sources_sha256": {name: sha256(root / name) for name in policy.SOURCE_FILES},
        }
        (root / policy.PLAN_FILENAME).write_text(json.dumps(plan), encoding="utf-8")
        summary, per_class = policy_tests.PromotionPolicyTests().fixture()
        summary.to_csv(output / "summary.csv", sep=";", index=False)
        per_class.to_csv(output / "metrics_per_class.csv", sep=";", index=False)
        decision = policy.choose_candidate(summary, per_class)
        pd.DataFrame(decision["promotion_diagnostics"]).to_csv(output / "gate_diagnostics.csv", sep=";", index=False)
        for name, value in (("promotion_decision.json", decision), ("design.json", {}), (policy.PLAN_FILENAME, plan)):
            (output / name).write_text(json.dumps(value), encoding="utf-8")
        (output / "oof_predictions.csv").write_text("fixture predictions", encoding="utf-8")
        experiment = {
            "state": "complete", "seed": 0, "LR_C": .25, "n_folds": 5,
            "sources_sha256": {name: sha256(root / name) for name in (*policy.SOURCE_FILES, policy.PLAN_FILENAME)},
            "artifacts_sha256": {p.name: sha256(p) for p in output.iterdir()},
            "promotion_decision": decision,
        }
        (output / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
        (output / "run_state.json").write_text('{"state":"complete"}', encoding="utf-8")
        return root, output

    def patched(self, root, output):
        return patch.multiple(classify, FINAL_EVALUATION_DIR=output, ROOT=root,
                              TRAIN_FILE=root / "train.csv", SELECTED_MODEL_FILE=root / "selected.json")

    def test_selection_and_standalone_load_without_cv_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.fixture(directory)
            with self.patched(root, output):
                selected = classify.record_selection()
                self.assertEqual(selected["schema_version"], 3)
                self.assertEqual(selected["candidate_id"], policy.INCUMBENT)
                for path in output.iterdir():
                    path.unlink()
                output.rmdir()
                self.assertEqual(classify.load_selection(), selected)

    def test_modified_source_or_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.fixture(directory)
            with self.patched(root, output):
                with (output / "metrics_per_class.csv").open("a", encoding="utf-8") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, "artifact"):
                    classify.record_selection()
                (root / "duplicate_exclusion.py").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "frozen decision plan"):
                    classify.record_selection()
                self.assertFalse((root / "selected.json").exists())

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.fixture(directory)
            (output / "run_state.json").write_text('{"state":"running"}', encoding="utf-8")
            with self.patched(root, output), self.assertRaisesRegex(ValueError, "incomplete"):
                classify.record_selection()

    def test_release_hash_detects_changed_inference_source_without_cv(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = self.fixture(directory)
            with self.patched(root, output):
                classify.record_selection()
                (root / "read_test_set.py").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed since selection"):
                    classify.load_selection()

    def test_legacy_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            path.write_text('{"schema_version":2,"candidate_id":"SVC+EXCL"}', encoding="utf-8")
            with patch.object(classify, "SELECTED_MODEL_FILE", path), self.assertRaisesRegex(ValueError, "Legacy"):
                classify.load_selection()

    def test_plan_requires_manual_review_and_current_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            path = root / policy.PLAN_FILENAME
            plan = policy.validate_decision_plan(path, root)
            for key, value in (("approved_to_fit", False), ("status", "draft")):
                bad = dict(plan, **{key: value})
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "not frozen/approved"):
                    policy.validate_decision_plan(path, root)
            path.write_text(json.dumps(plan), encoding="utf-8")
            (root / "review.json").write_text("different examples", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reviewed augmentation"):
                policy.validate_decision_plan(path, root)


if __name__ == "__main__":
    unittest.main()
