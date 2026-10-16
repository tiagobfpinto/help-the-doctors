"""Target-free exclusion ranking, fallback, and a counterexample to its premise."""

import unittest

import numpy as np

from audit_duplicate_exclusion import exclude_seen_classes, training_note_labels


class DuplicateExclusionTests(unittest.TestCase):
    def test_all_seen_classes_excluded_and_next_best_is_selected(self):
        seen = training_note_labels(["The note", "The note", "The note"], ["A", "B", "C"])
        raw, ruled, detail = exclude_seen_classes([[8, 9, 7, 6, 5]], list("ABCDE"), ["the NOTE"], seen)
        self.assertEqual(raw.tolist(), ["B"])
        self.assertEqual(ruled.tolist(), ["D"])
        self.assertEqual(detail.n_excluded_classes.iloc[0], 3)
        self.assertEqual(detail.rule_choice_rank_before_exclusion.iloc[0], 4)

    def test_empty_notes_never_match_and_no_copy_stays_raw(self):
        seen = training_note_labels(["", "  ", "other"], ["A", "B", "A"])
        self.assertNotIn("", seen)
        raw, ruled, detail = exclude_seen_classes([[2, 1], [1, 2]], ["A", "B"], ["", "new"], seen)
        np.testing.assert_array_equal(raw, ruled)
        self.assertFalse(detail.has_train_copy.any())

    def test_total_exclusion_falls_back_to_raw(self):
        raw, ruled, detail = exclude_seen_classes([[1, 4]], ["A", "B"], ["x"], {"x": {"A", "B"}})
        self.assertEqual(raw.tolist(), ruled.tolist())
        self.assertTrue(detail.fallback_all_excluded.iloc[0])
        self.assertFalse(detail.changed.iloc[0])

    def test_whitespace_and_case_are_normalized(self):
        seen = training_note_labels([" The\n same   NOTE "], ["A"])
        _, ruled, detail = exclude_seen_classes([[3, 2]], ["A", "B"], ["the same note"], seen)
        self.assertEqual(ruled.tolist(), ["B"])
        self.assertTrue(detail.has_train_copy.iloc[0])

    def test_rule_can_harm_when_same_note_reuses_its_class(self):
        # The held-out target A is used only to assess the output, never by the rule.
        _, ruled, _ = exclude_seen_classes([[3, 1]], ["A", "B"], ["note"], {"note": {"A"}})
        self.assertNotEqual(ruled[0], "A")

    def test_lookup_from_training_subset_does_not_include_held_out_class(self):
        notes, labels = ["same note", "same note"], ["A", "B"]
        training_indices = [0]
        seen = training_note_labels([notes[i] for i in training_indices], [labels[i] for i in training_indices])
        self.assertEqual(seen, {"same note": {"A"}})
        # Including the held-out B label would block both classes and fall back
        # to A. A fold-local lookup instead allows B, without reading its target.
        _, ruled, detail = exclude_seen_classes([[3, 2]], ["A", "B"], [notes[1]], seen)
        self.assertEqual(ruled.tolist(), ["B"])
        self.assertFalse(detail.fallback_all_excluded.iloc[0])


if __name__ == "__main__":
    unittest.main()
