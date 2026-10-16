"""A stricter key must distinguish boilerplate and keep useful title-only input."""

import unittest

import pandas as pd

from audit_duplicate_key_sensitivity import pair_keys


class PairKeySensitivityTests(unittest.TestCase):
    def test_same_boilerplate_under_distinct_titles_is_not_same_input(self):
        keys = pair_keys(pd.DataFrame({
            "sample_name": ["Shoulder Contusion", "Hip Pain"],
            "transcription": ["CHIEF COMPLAINT:", "CHIEF COMPLAINT:"],
        }))
        self.assertNotEqual(keys.iloc[0], keys.iloc[1])

    def test_title_only_inputs_match_after_cleaning_and_wholly_empty_inputs_skip(self):
        keys = pair_keys(pd.DataFrame({
            "sample_name": ["The title", " the   TITLE ", ""],
            "transcription": ["", "", " "],
        }))
        self.assertEqual(keys.iloc[0], keys.iloc[1])
        self.assertNotEqual(keys.iloc[0], "")
        self.assertEqual(keys.iloc[2], "")


if __name__ == "__main__":
    unittest.main()
