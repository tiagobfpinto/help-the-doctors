"""Exercise the inference entrypoint, ordering, decoder and output alignment."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import classify
from duplicate_exclusion import fit_exclusion_index


class FinalInferenceTests(unittest.TestCase):
    def test_model_and_index_are_fitted_before_test_and_decoder_writes_aligned_labels(self):
        train = [
            {"record_id": 1, "medical_specialty": "Orthopedic", "description": "",
             "sample_name": "Radius", "transcription": "Fracture fixation"},
            {"record_id": 2, "medical_specialty": "Surgery", "description": "",
             "sample_name": "Different", "transcription": "Another case"},
        ]
        cases = [
            {"case_id": 1, "description": "", "sample_name": " radius ",
             "transcription": " fracture   FIXATION "},
            {"case_id": 2, "description": "", "sample_name": "Unseen",
             "transcription": "Unseen note"},
        ]
        selection = {"candidate_id": "SVC+EXCL", "model_id": "SVC",
                     "model": "LinearSVC", "decoder": "exclude_seen"}
        events = []
        fitted = Mock()
        fitted.classes_ = np.array(["Dermatology", "Orthopedic", "Surgery"])
        fitted.decision_function.return_value = np.array([[0., 3., 2.], [2., 1., 0.]])
        batch = SimpleNamespace(y=pd.Series([row["medical_specialty"] for row in train]), metadata={})

        def fit(frame, model_id, **kwargs):
            self.assertEqual(len(frame), 2)
            self.assertEqual(model_id, "SVC")
            events.append("fit")
            return fitted, batch

        def build_index(frame):
            self.assertEqual(events, ["fit"])
            events.append("index")
            return fit_exclusion_index(frame)

        def read_test(path):
            self.assertEqual(events, ["fit", "index"])
            events.append("test")
            return cases

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test.csv"
            source.write_text("unchanged test fixture", encoding="utf-8")
            output = root / "results.txt"
            with patch.multiple(classify, FINAL_OUTPUT_DIR=root / "final",
                                RESULTS_FILE=output, TEST_FILE=source), \
                 patch.object(classify, "record_selection", return_value=selection), \
                 patch.object(classify, "repair_training_csv", return_value=(train, [])), \
                 patch.object(classify, "fit_evaluated_model", side_effect=fit), \
                 patch.object(classify, "fit_exclusion_index", side_effect=build_index), \
                 patch.object(classify, "read_test_cases", side_effect=read_test):
                classify.main(["--select"])
            self.assertEqual(output.read_text(encoding="utf-8").splitlines(),
                             ["Surgery", "Dermatology"])
            meta = json.loads((root / "final/inference.json").read_text(encoding="utf-8"))
            self.assertIsNone(meta["test_metrics"])
            self.assertEqual(meta["results_lines"], len(cases))
            self.assertEqual(meta["exclusion_index_original_training_rows"], len(train))
            diagnostics = pd.read_csv(root / "final/decoding_diagnostics.csv", sep=";")
            self.assertEqual(diagnostics.case_id.tolist(), [1, 2])
            self.assertEqual(diagnostics.changed.tolist(), [True, False])
            self.assertEqual(source.read_text(encoding="utf-8"), "unchanged test fixture")

    def test_raw_candidate_preserves_classifier_ranking_despite_matching_index(self):
        frame = pd.DataFrame([{"sample_name": "A", "transcription": "B",
                               "medical_specialty": "Orthopedic"}])
        index = fit_exclusion_index(frame)
        model = Mock()
        model.classes_ = np.array(["Orthopedic", "Surgery", "Dermatology"])
        model.decision_function.return_value = np.array([[3., 2., 1.]])
        labels, result = classify.predict_selected(
            model, index, frame.drop(columns="medical_specialty"),
            {"model_id": "SVC", "decoder": "raw"},
        )
        self.assertEqual(labels, ["Orthopedic"])
        self.assertEqual(result.predictions.tolist(), ["Surgery"])


if __name__ == "__main__":
    unittest.main()
