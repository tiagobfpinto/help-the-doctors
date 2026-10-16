"""Checks for source isolation, protected text, matched controls and weight mass."""

import re
import unittest

import numpy as np
import pandas as pd

from augmentation import (
    ARTICLE, CONDITIONS, LABEL, article_dropout, clean_inputs,
    fit_evaluated_model, identity_groups, prediction_input, prepare_training,
)


def sample_train():
    rows = []
    for i, label in enumerate(["Dermatology", "Dermatology", "Surgery", "Surgery", "Surgery"]):
        rows.append({
            "record_id": i + 1, LABEL: label, "sample_name": f"The procedure {i}",
            "description": f"Description for procedure {i}",
            "transcription": "The patient has no fever. The dose is 2.5 mg. The left eye was normal.",
        })
    return pd.DataFrame(rows)


class AugmentationTests(unittest.TestCase):
    def test_only_articles_are_removed_and_clinical_tokens_survive(self):
        text = "The patient denies the pain; no fever, 2.5 mg, hepatitis A, without rash in the left eye."
        result = article_dropout(text, seed=0, fold=1, record_id=1)
        self.assertEqual(len(result.removed_spans), 1)
        tokens = lambda value: [t for t in re.findall(r"(?u)\b\w+\b", value) if t.lower() != "the"]
        self.assertEqual(tokens(text), tokens(result.text))
        self.assertIn("no fever", result.text)
        self.assertIn("2.5 mg", result.text)
        self.assertIn("hepatitis A", result.text)
        self.assertIn("without rash", result.text)

    def test_rate_counts_articles_not_all_words_and_uses_floor(self):
        result = article_dropout("the word " * 11, seed=0, fold=2, record_id=9)
        self.assertEqual(result.article_count, 11)
        self.assertEqual(result.original_word_count, 22)
        self.assertEqual(len(result.removed_spans), 2)
        self.assertEqual(len(ARTICLE.findall(result.text)), 9)

    def test_source_variants_are_deterministic_and_rejections_explicit(self):
        kwargs = {"seed": 0, "fold": 2, "record_id": 12}
        text = "the patient and the clinician discussed the treatment with the nurse"
        self.assertEqual(article_dropout(text, **kwargs), article_dropout(text, **kwargs))
        self.assertEqual(article_dropout("fever absent", **kwargs).rejection, "no_article")
        self.assertEqual(article_dropout("the", **kwargs).rejection, "empty_after_deletion")

    def test_matched_copies_share_sources_labels_counts_and_weights(self):
        original = sample_train()
        preserved = original.copy(deep=True)
        dropped = prepare_training(original, "D", seed=0, fold=1)
        copied = prepare_training(original, "E", seed=0, fold=1)
        pd.testing.assert_frame_equal(original, preserved)
        self.assertEqual(dropped.children["parent_record_id"].tolist(), [1, 2])
        self.assertEqual(dropped.metadata["added_rows"], copied.metadata["added_rows"])
        pd.testing.assert_series_equal(dropped.y, copied.y)
        np.testing.assert_allclose(dropped.sample_weight, copied.sample_weight)
        self.assertTrue(dropped.X["sample_name"].iloc[5:].reset_index(drop=True).equals(
            clean_inputs(original)["sample_name"].iloc[:2].reset_index(drop=True)))
        self.assertNotEqual(dropped.X["transcription"].iloc[-1], copied.X["transcription"].iloc[-1])
        self.assertEqual(copied.X["transcription"].iloc[-1], original["transcription"].iloc[1])

    def test_weight_mass_fixed_and_class_weights_come_only_from_originals(self):
        original = sample_train()
        for condition in CONDITIONS:
            batch = prepare_training(original, condition, seed=0, fold=1)
            self.assertAlmostEqual(batch.sample_weight.sum(), len(original))
            self.assertEqual(len(batch.X), len(batch.y))
            self.assertEqual(len(batch.y), len(batch.sample_weight))
        batch = prepare_training(original, "D")
        self.assertAlmostEqual(batch.metadata["class_weights_from_originals"]["Dermatology"], 5 / 4)
        self.assertAlmostEqual(batch.metadata["class_weights_from_originals"]["Surgery"], 5 / 6)
        self.assertGreater(batch.metadata["weight_mass_by_class"]["Dermatology"], 2.5)

    def test_groups_merge_notes_and_fallback_titles_but_not_all_empty_inputs(self):
        frame = pd.DataFrame({
            "record_id": range(1, 7),
            "sample_name": ["first", "second", "The title", " the   TITLE ", "", ""],
            "transcription": ["The note", " the\nNOTE ", "", "", "", ""],
        })
        groups = identity_groups(frame)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[4], groups[5])

    def test_all_evaluated_families_fit_with_their_prediction_input(self):
        train = sample_train()
        for model_id in ["B", "D", "E", "NB", "SVC"]:
            with self.subTest(model_id=model_id):
                model, batch = fit_evaluated_model(train, model_id, seed=0, fold=1)
                prediction = model.predict(prediction_input(train, model_id))
                self.assertEqual(len(prediction), len(train))
                self.assertEqual(set(model.classes_), set(train[LABEL]))
                self.assertAlmostEqual(batch.sample_weight.sum(), len(train))


if __name__ == "__main__":
    unittest.main()
