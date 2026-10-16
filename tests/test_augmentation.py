"""Section-mixture isolation and matched controls; these tests never fit models.

Former article-deletion tests are preserved in the pre-change snapshot.
"""

import json
import unittest

import numpy as np
import pandas as pd

from augmentation import (
    CONDITIONS, EVALUATED_MODEL_IDS, LABEL, MIN_SECTION_WORDS,
    build_evaluated_model, clean_inputs, identity_groups, parse_sections,
    prediction_input, prepare_training,
)


def sample_train():
    notes = [
        "HISTORY: The adult patient reports a red rash across both arms today.,"
        "ASSESSMENT: The rash is consistent with inflammation and requires further observation.,"
        "PLAN: Continue topical treatment and arrange another examination after two weeks.",
        "HISTORY: A young child developed itching over the chest after outdoor exposure.,"
        "ASSESSMENT: This child has a persistent skin eruption without signs of infection.,"
        "PLAN: Begin the prescribed ointment and return for reassessment in one month.",
    ]
    return pd.DataFrame([{
        "record_id": i + 1, LABEL: label, "sample_name": f"Procedure {i}",
        "description": f"Description {i}", "transcription": notes[i % 2],
    } for i, label in enumerate(["Dermatology", "Dermatology", "Surgery", "Surgery", "Surgery"])])


class SectionRecombinationTests(unittest.TestCase):
    def test_parser_recognizes_real_allowlisted_headings_and_whole_bodies(self):
        text = "HISTORY: The note contains eight useful clinical words about this patient.,PLAN: Follow up tomorrow."
        sections = parse_sections(text)
        self.assertEqual([s.section_type for s in sections], ["HISTORY", "PLAN"])
        self.assertEqual(sections[0].body, "The note contains eight useful clinical words about this patient.")
        for section in sections:
            self.assertEqual(text[section.body_start:section.body_end], section.body)
        self.assertGreaterEqual(sections[0].word_count, MIN_SECTION_WORDS)

    def test_arbitrary_uppercase_or_inline_prose_is_not_a_header(self):
        self.assertEqual(parse_sections("ARBITRARY CONTENT: One two three four five six seven eight."), ())
        self.assertEqual(parse_sections("The prose mentions HISTORY: without a section boundary."), ())
        self.assertEqual(parse_sections("history: lowercase prose is not an uppercase heading."), ())

    def test_treatment_and_nested_exam_headings_close_the_previous_body(self):
        text = "IMPRESSION: Acne with folliculitis.,TREATMENT: Continue treatment with the usual instructions and return next month."
        sections = parse_sections(text)
        self.assertEqual(sections[0].body, "Acne with folliculitis.")
        self.assertEqual(sections[1].section_type, "TREATMENT")
        nested = parse_sections("PHYSICAL EXAMINATION:GENERAL: Awake alert and well appearing. ,PLAN: Return tomorrow.")
        self.assertEqual([s.section_type for s in nested], ["PHYSICAL EXAMINATION", "GENERAL", "PLAN"])
        self.assertEqual(nested[0].word_count, 0)

    def test_repeated_heading_does_not_match_a_short_title_to_a_long_narrative(self):
        train = sample_train().iloc[:2].copy()
        train["transcription"] = (
            "PROCEDURE: First procedure title with more than eight whole words here.,"
            "PLAN: Return tomorrow.,"
            "PROCEDURE: Longer procedural account with more than eight different words provided."
        )
        train.loc[1, "transcription"] = train.loc[1, "transcription"].replace("First", "Second")
        batch = prepare_training(train, "D")
        self.assertTrue(batch.children.empty)
        self.assertTrue(batch.audit.rejection.eq("no_unambiguous_parent_section_at_min_words").all())

    def test_unrecognized_internal_heading_makes_a_body_ineligible(self):
        train = sample_train().iloc[:2].copy()
        train["transcription"] = (
            "HISTORY: These are more than eight words describing a clinical history.,"
            "ARBITRARY CONTENT: These extra sentences must not be swallowed into a history section.,PLAN: Return."
        )
        train.loc[1, "transcription"] = train.loc[1, "transcription"].replace("These are", "Those were")
        batch = prepare_training(train, "D")
        self.assertTrue(batch.children.empty)
        self.assertTrue(batch.audit.rejection.eq("no_unambiguous_parent_section_at_min_words").all())

    def test_exactly_one_complete_body_is_replaced_and_other_characters_survive(self):
        train = sample_train()
        before = train.copy(deep=True)
        batch = prepare_training(train, "D", seed=0, fold=1)
        pd.testing.assert_frame_equal(train, before)
        originals = train.set_index("record_id")
        for child in batch.children.to_dict("records"):
            item = batch.audit.loc[batch.audit.parent_record_id.eq(child["parent_record_id"])].iloc[0]
            parent = originals.loc[child["parent_record_id"]]
            donor = originals.loc[child["donor_record_id"]]
            start, end = json.loads(item.parent_body_span)
            ds, de = json.loads(item.donor_body_span)
            expected = parent.transcription[:start] + donor.transcription[ds:de] + parent.transcription[end:]
            self.assertEqual(child["transcription"], expected)
            self.assertEqual(child["sample_name"], parent.sample_name)
            self.assertEqual(child[LABEL], donor[LABEL])
            self.assertNotEqual(item.parent_group_id, item.donor_group_id)
            self.assertGreaterEqual(item.removed_words, MIN_SECTION_WORDS)
            self.assertGreaterEqual(item.added_words, MIN_SECTION_WORDS)
            self.assertNotIn("INDEPENDENT DONOR", child["transcription"])

    def test_only_original_rows_supplied_for_this_fold_can_supply_either_source(self):
        all_rows = sample_train()
        held_out = all_rows.iloc[[1]]
        training = all_rows.drop(index=1)
        no_donor = prepare_training(training, "D", seed=0, fold=1)
        self.assertTrue(no_donor.children.empty)
        self.assertIn("no_distinct_same_class_original_donor", no_donor.audit.rejection.tolist())
        full_batch = prepare_training(all_rows, "D", seed=0, fold=1)
        self.assertTrue(set(full_batch.children.donor_record_id) <= set(all_rows.record_id))
        self.assertTrue(set(no_donor.children.donor_record_id).isdisjoint(set(held_out.record_id)))

    def test_child_frames_are_rejected_as_original_source_pools(self):
        children = prepare_training(sample_train(), "D").children
        with self.assertRaisesRegex(ValueError, "Only original"):
            prepare_training(children, "D")

    def test_different_specialties_cannot_donate_even_when_headings_match(self):
        batch = prepare_training(sample_train().iloc[[0, 3]], "D")
        self.assertTrue(batch.children.empty)
        self.assertEqual(batch.audit.rejection.tolist(), ["no_distinct_same_class_original_donor"])

    def test_exact_duplicate_note_groups_cannot_donate_to_each_other(self):
        train = sample_train().iloc[:2].copy()
        train.loc[train.index[1], "transcription"] = train.iloc[0].transcription
        batch = prepare_training(train, "D")
        self.assertTrue(batch.children.empty)
        self.assertTrue(batch.audit.rejection.eq("no_distinct_same_class_original_donor").all())

    def test_notes_without_two_nonempty_real_section_types_are_skipped(self):
        for text in ("", "Arbitrary text without headings.", "HISTORY: Only one recognized heading with enough words to be a narrative.", "HISTORY: ,PLAN:"):
            with self.subTest(text=text):
                train = sample_train().iloc[:2].copy()
                train["transcription"] = text
                batch = prepare_training(train, "D")
                self.assertTrue(batch.children.empty)
                self.assertFalse(batch.audit.changed.any())

    def test_donor_must_have_same_heading_type_and_different_body(self):
        train = sample_train().iloc[:2].copy()
        train.loc[1, "transcription"] = train.loc[1, "transcription"].replace("HISTORY:", "FINDINGS:").replace("ASSESSMENT:", "IMPRESSION:").replace("PLAN:", "RECOMMENDATIONS:")
        batch = prepare_training(train, "D")
        self.assertTrue(batch.children.empty)
        self.assertTrue(batch.audit.rejection.eq("no_matching_different_section_at_min_words").all())

    def test_an_identical_eligible_body_is_not_accepted_as_recombination(self):
        train = sample_train().iloc[:2].copy()
        common = "HISTORY: The same original clinical narrative has more than eight words here.,PLAN: "
        train.loc[0, "transcription"] = common + "Return."
        train.loc[1, "transcription"] = common + "Rest."
        self.assertNotEqual(identity_groups(train).iloc[0], identity_groups(train).iloc[1])
        batch = prepare_training(train, "D")
        self.assertTrue(batch.children.empty)
        self.assertTrue(batch.audit.rejection.eq("no_matching_different_section_at_min_words").all())

    def test_determinism_does_not_depend_on_input_row_order(self):
        train = sample_train()
        first = prepare_training(train, "D", seed=0, fold=1)
        other = prepare_training(train.sample(frac=1, random_state=7), "D", seed=0, fold=1)
        columns = ["parent_record_id", "donor_record_id", "section_type", "transcription"]
        pd.testing.assert_frame_equal(first.children[columns].sort_values("parent_record_id").reset_index(drop=True),
                                      other.children[columns].sort_values("parent_record_id").reset_index(drop=True))

    def test_matched_copies_share_accepted_parents_labels_counts_and_weights(self):
        train = sample_train()
        mixed, copied = (prepare_training(train, c, seed=0, fold=1) for c in ("D", "E"))
        self.assertEqual(mixed.children.parent_record_id.tolist(), [1, 2])
        pd.testing.assert_frame_equal(mixed.children, copied.children)
        pd.testing.assert_series_equal(mixed.y, copied.y)
        np.testing.assert_array_equal(mixed.sample_weight, copied.sample_weight)
        originals = clean_inputs(train).set_axis(train.record_id)
        for offset, parent_id in enumerate(copied.children.parent_record_id, start=len(train)):
            self.assertEqual(copied.X.iloc[offset].to_dict(), originals.loc[parent_id].to_dict())
        self.assertTrue((mixed.X.transcription.iloc[len(train):].to_numpy() != copied.X.transcription.iloc[len(train):].to_numpy()).all())

    def test_weight_mass_and_original_class_weight_estimation_are_unchanged(self):
        train = sample_train()
        for condition in CONDITIONS:
            batch = prepare_training(train, condition)
            self.assertAlmostEqual(batch.sample_weight.sum(), len(train))
            self.assertEqual(len(batch.X), len(batch.y))
        batch = prepare_training(train, "D")
        self.assertAlmostEqual(batch.metadata["class_weights_from_originals"]["Dermatology"], 5 / 4)
        self.assertGreater(batch.metadata["weight_mass_by_class"]["Dermatology"], 2.5)
        self.assertTrue(batch.metadata["no_article_deletion"])
        self.assertTrue(batch.metadata["no_markers_in_model_input"])

    def test_model_registry_and_input_api_construct_without_fitting(self):
        self.assertEqual(EVALUATED_MODEL_IDS, ("NB", "B", "D", "E", "SVC", "SVC_AUG", "SVC_COPY"))
        for model_id in EVALUATED_MODEL_IDS:
            estimator = build_evaluated_model(model_id)
            self.assertFalse(hasattr(estimator.named_steps["classifier"], "classes_"))
            self.assertEqual(len(prediction_input(sample_train(), model_id)), len(sample_train()))
        for model_id in ("SVC", "SVC_AUG", "SVC_COPY"):
            self.assertEqual(type(build_evaluated_model(model_id).named_steps["classifier"]).__name__, "LinearSVC")

    def test_group_identity_rules_still_handle_missing_inputs(self):
        frame = pd.DataFrame({"record_id": range(1, 7),
                              "sample_name": ["first", "second", "The title", " the TITLE ", "", ""],
                              "transcription": ["The note", " the\nNOTE ", "", "", "", ""]})
        groups = identity_groups(frame)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[4], groups[5])


if __name__ == "__main__":
    unittest.main()
