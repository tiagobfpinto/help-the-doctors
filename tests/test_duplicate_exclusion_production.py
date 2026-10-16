"""Shared decoder contract: targets never read; originals only; stable fallbacks."""

import unittest

import numpy as np
import pandas as pd

from duplicate_exclusion import exact_input_keys, fit_exclusion_index, score_model


def rows(titles, notes, labels=None):
    frame = pd.DataFrame({"sample_name": titles, "transcription": notes})
    if labels is not None:
        frame["medical_specialty"] = labels
    return frame


class ProductionDecoderTests(unittest.TestCase):
    def test_multiple_labels_excluded_and_validation_targets_ignored(self):
        index = fit_exclusion_index(rows(["same"] * 3, ["note"] * 3, ["A", "B", "C"]))
        val = rows([" SAME "], [" NOTE "], ["a deliberately irrelevant validation target"])
        result = index.predict([[9, 8, 7, 6]], list("ABCD"), val)
        self.assertEqual(result.raw_predictions.tolist(), ["A"])
        self.assertEqual(result.predictions.tolist(), ["D"])
        self.assertEqual(result.diagnostics.decoded_choice_raw_rank.iloc[0], 4)
        self.assertEqual(result.diagnostics.n_excluded_classes.iloc[0], 3)

    def test_boilerplate_with_distinct_titles_does_not_match(self):
        index = fit_exclusion_index(rows(["Shoulder Contusion"], ["CHIEF COMPLAINT:"], ["A"]))
        result = index.predict([[5, 1]], ["A", "B"], rows(["Hip Pain"], ["CHIEF COMPLAINT:"]))
        self.assertEqual(result.predictions.tolist(), ["A"])
        self.assertFalse(result.diagnostics.has_train_match.iloc[0])

    def test_title_only_matches_but_wholly_empty_does_not(self):
        index = fit_exclusion_index(rows(["Only title", ""], ["", ""], ["A", "A"]))
        result = index.predict([[5, 1], [5, 1]], ["A", "B"], rows([" only  TITLE ", ""], ["", ""]))
        self.assertEqual(result.predictions.tolist(), ["B", "A"])
        self.assertEqual(result.diagnostics.input_available.tolist(), [True, False])

    def test_all_blocked_fallback_and_unseen_input(self):
        index = fit_exclusion_index(rows(["title", "title"], ["note", "note"], ["A", "B"]))
        result = index.predict([[1, 2], [1, 2]], ["A", "B"], rows(["title", "different"], ["note", "new"]))
        self.assertEqual(result.predictions.tolist(), ["B", "B"])
        self.assertEqual(result.diagnostics.fallback_all_excluded.tolist(), [True, False])
        self.assertFalse(result.diagnostics.changed.any())

    def test_index_contains_only_originals_supplied_by_caller(self):
        original = rows(["title"], ["note"], ["A"])
        index = fit_exclusion_index(original)
        # A hypothetical descendant with class B is intentionally not supplied.
        result = index.predict([[3, 2, 1]], list("ABC"), rows(["title"], ["note"]))
        self.assertEqual(index.original_training_rows, 1)
        self.assertEqual(result.predictions.tolist(), ["B"])

    def test_input_keys_ignore_whitespace_and_case(self):
        keys = exact_input_keys(rows([" A  title ", "a title"], [" note\ntext ", "NOTE TEXT"]))
        self.assertEqual(keys.iloc[0], keys.iloc[1])

    def test_score_model_uses_nb_log_probability_and_dataframe_for_linear_models(self):
        class ToyModel:
            classes_ = np.array(["A", "B"])

            def predict_log_proba(self, data):
                assert isinstance(data, pd.Series)
                return [[-1.0, -2.0]]

            def decision_function(self, data):
                assert isinstance(data, pd.DataFrame)
                return [[2.0, 1.0]]

        frame = rows(["t"], ["n"])
        frame["description"] = "d"
        model = ToyModel()
        np.testing.assert_array_equal(score_model(model, frame, "NB")[0], [[-1, -2]])
        np.testing.assert_array_equal(score_model(model, frame, "SVC")[0], [[2, 1]])


if __name__ == "__main__":
    unittest.main()
