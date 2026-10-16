"""Promotion must require both gains, both references and original-only gates."""
import unittest

import pandas as pd

import augmentation_policy as policy


class PromotionPolicyTests(unittest.TestCase):
    def fixture(self):
        summary, per_class = [], []
        for protocol in policy.PROTOCOLS:
            for mid in policy.MODEL_IDS:
                for decoder in policy.DECODERS:
                    cid = mid + ("+EXCL" if decoder == "exclude_seen" else "")
                    summary.append({"candidate_id": cid, "model_id": mid, "decoder": decoder,
                                    "model": policy.MODEL_NAMES[mid], "protocol": protocol,
                                    "seed": 0, "accuracy": .78, "f1_macro": .70, "min_f1": .50})
                    for label, f1 in [("Dermatology", .50)] + [(f"Class{x}", .7181818181818181) for x in range(11)]:
                        per_class.append({"protocol": protocol, "seed": 0, "candidate_id": cid,
                                          "class": label, "f1": f1})
        return pd.DataFrame(summary), pd.DataFrame(per_class)

    def set_metrics(self, tables, candidate, *, macro=None, derm=None, accuracy=None, protocol=None):
        summary, pc = tables
        protocol = protocol or policy.PRIMARY_PROTOCOL
        smask = summary.candidate_id.eq(candidate) & summary.protocol.eq(protocol)
        pmask = pc.candidate_id.eq(candidate) & pc.protocol.eq(protocol)
        current_macro = float(summary.loc[smask, "f1_macro"].iloc[0])
        current_derm = float(pc.loc[pmask & pc["class"].eq("Dermatology"), "f1"].iloc[0])
        macro = current_macro if macro is None else macro
        derm = current_derm if derm is None else derm
        pc.loc[pmask & pc["class"].eq("Dermatology"), "f1"] = derm
        pc.loc[pmask & ~pc["class"].eq("Dermatology"), "f1"] = (12 * macro - derm) / 11
        summary.loc[smask, "f1_macro"] = macro
        summary.loc[smask, "min_f1"] = pc.loc[pmask, "f1"].min()
        if accuracy is not None:
            summary.loc[smask, "accuracy"] = accuracy

    def choose(self, tables):
        return policy.choose_candidate(*tables)

    def test_equal_scores_retain_incumbent(self):
        selected = self.choose(self.fixture())
        self.assertEqual(selected["candidate_id"], "SVC+EXCL")
        self.assertFalse(selected["promotion_applied"])

    def test_joint_macro_and_dermatology_gain_is_required(self):
        for macro, derm in ((.71, .50), (.70, .51), (.71, .49), (.69, .51)):
            tables = self.fixture()
            self.set_metrics(tables, "D+EXCL", macro=macro, derm=derm)
            self.assertEqual(self.choose(tables)["candidate_id"], policy.INCUMBENT)
        tables = self.fixture()
        self.set_metrics(tables, "D+EXCL", macro=.71, derm=.51)
        self.assertEqual(self.choose(tables)["candidate_id"], "D+EXCL")

    def test_incumbent_gate_prevents_lr_only_gain(self):
        tables = self.fixture()
        self.set_metrics(tables, "B+EXCL", macro=.60, derm=.40)
        self.set_metrics(tables, "D+EXCL", macro=.65, derm=.45)
        self.assertEqual(self.choose(tables)["candidate_id"], policy.INCUMBENT)

    def test_svc_gate_is_against_own_family(self):
        tables = self.fixture()
        self.set_metrics(tables, "B+EXCL", macro=.60, derm=.40)
        self.set_metrics(tables, "SVC_AUG+EXCL", macro=.65, derm=.45)
        self.assertEqual(self.choose(tables)["candidate_id"], policy.INCUMBENT)

    def test_one_percentage_point_boundary_is_inclusive(self):
        tables = self.fixture()
        self.set_metrics(tables, "D+EXCL", macro=.71, derm=.51, accuracy=.77)
        self.assertEqual(self.choose(tables)["candidate_id"], "D+EXCL")
        self.set_metrics(tables, "D+EXCL", accuracy=.769)
        self.assertEqual(self.choose(tables)["candidate_id"], policy.INCUMBENT)

    def test_absolute_class_constraint_is_not_overridden_by_average(self):
        tables = self.fixture()
        self.set_metrics(tables, "D+EXCL", macro=.71, derm=.51)
        summary, pc = tables
        mask = pc.protocol.eq(policy.PRIMARY_PROTOCOL) & pc.candidate_id.eq("D+EXCL")
        pc.loc[mask & pc["class"].eq("Class0"), "f1"] = .24
        summary.loc[summary.protocol.eq(policy.PRIMARY_PROTOCOL) & summary.candidate_id.eq("D+EXCL"), ["f1_macro", "min_f1"]] = [pc.loc[mask, "f1"].mean(), .24]
        self.assertEqual(self.choose(tables)["candidate_id"], policy.INCUMBENT)

    def test_copy_raw_and_grouped_cannot_promote(self):
        tables = self.fixture()
        for candidate in ("E+EXCL", "SVC_COPY+EXCL", "D", "SVC_AUG"):
            self.set_metrics(tables, candidate, macro=.90, derm=.90, accuracy=.99)
        self.set_metrics(tables, "D+EXCL", macro=.95, derm=.95, accuracy=.99, protocol=policy.GROUPED_PROTOCOL)
        self.assertEqual(self.choose(tables)["candidate_id"], policy.INCUMBENT)

    def test_rank_is_macro_then_dermatology_then_accuracy(self):
        tables = self.fixture()
        self.set_metrics(tables, "D+EXCL", macro=.72, derm=.52, accuracy=.78)
        self.set_metrics(tables, "SVC_AUG+EXCL", macro=.71, derm=.60, accuracy=.90)
        self.assertEqual(self.choose(tables)["candidate_id"], "D+EXCL")
        self.set_metrics(tables, "SVC_AUG+EXCL", macro=.72, derm=.53)
        self.assertEqual(self.choose(tables)["candidate_id"], "SVC_AUG+EXCL")

    def test_incomplete_class_and_summary_reports_are_rejected(self):
        summary, per_class = self.fixture()
        for a, b in ((summary.iloc[:-1], per_class), (summary, per_class.iloc[:-1]),
                     (pd.concat([summary, summary.iloc[:1]]), per_class)):
            with self.assertRaises(ValueError):
                policy.choose_candidate(a, b)
        summary.loc[0, "f1_macro"] = .99
        with self.assertRaisesRegex(ValueError, "per-class"):
            policy.choose_candidate(summary, per_class)


if __name__ == "__main__":
    unittest.main()
