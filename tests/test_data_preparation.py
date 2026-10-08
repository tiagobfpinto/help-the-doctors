"""Verificações das regras que podem recuperar ou deslocar texto entre casos."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

from data_preparation import COLUMNS, DatasetFormatError, audit_records, repair_training_csv


def csv_line(fields):
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=";", lineterminator="\n").writerow(fields)
    return buffer.getvalue()


class RepairTests(unittest.TestCase):
    def repair(self, *lines):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.csv"
            source.write_text(csv_line(COLUMNS) + "".join(lines), encoding="utf-8")
            return repair_training_csv(source)

    def test_quoted_semicolon_is_content(self):
        fields = ["Surgery", "descrição; com separador", "nome", 'Texto com "aspas"', ""]
        records, changes = self.repair(csv_line(fields))
        self.assertEqual([records[0][field] for field in COLUMNS], fields)
        self.assertEqual(changes, [])

    def test_unclosed_description_does_not_swallow_next_case(self):
        records, changes = self.repair(
            'Surgery;"Descrição parcial\n',
            csv_line(["Neurology", "descrição", "nome", "texto", ""]),
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["description"], "Descrição parcial")
        self.assertEqual(records[0]["transcription"], "")
        self.assertTrue(records[0]["is_partial"])
        self.assertEqual(records[1]["medical_specialty"], "Neurology")
        self.assertEqual(changes[0]["action"], "close_truncated_description")

    def test_continuation_preserves_available_text_and_source_lines(self):
        records, changes = self.repair(
            csv_line(["General Medicine", "desc", "nome", 'Parte um"', "", "", ""]),
            csv_line(["continuação", "", "", "", ""]),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["transcription"], 'Parte um"\ncontinuação')
        self.assertEqual((records[0]["source_line_start"], records[0]["source_line_end"]), (2, 3))
        self.assertEqual(len(changes), 2)

    def test_nonempty_extra_field_is_not_discarded(self):
        with self.assertRaises(DatasetFormatError):
            self.repair(csv_line(["Surgery", "d", "n", "t", "k", "conteúdo", ""]))

    def test_unexplained_multiline_case_is_not_guessed(self):
        with self.assertRaises(DatasetFormatError):
            self.repair('Surgery;"Descrição; ambígua\n')

    def test_orphan_continuation_is_not_assigned_a_label(self):
        with self.assertRaises(DatasetFormatError):
            self.repair(csv_line(["continuação", "", "", "", ""]))

    def test_continuation_after_normal_case_is_rejected(self):
        with self.assertRaises(DatasetFormatError):
            self.repair(csv_line(["Surgery", "d", "n", "t", ""]),
                        csv_line(["continuação", "", "", "", ""]))

    def test_blank_transcriptions_are_not_duplicate_groups(self):
        records, _ = self.repair(
            csv_line(["Surgery", "d", "n", "", ""]),
            csv_line(["Neurology", "d", "n", "", ""]),
        )
        audit = audit_records(records)
        self.assertTrue(all(row["transcription_empty"] for row in audit))
        self.assertTrue(all(not row["duplicate_group_id"] for row in audit))

    def test_conflicting_labels_are_reported_and_preserved(self):
        records, _ = self.repair(
            csv_line(["Surgery", "d", "n", "Mesmo texto", ""]),
            csv_line(["Neurology", "d", "n", "Mesmo texto", ""]),
        )
        audit = audit_records(records)
        self.assertTrue(all(row["conflicting_labels"] for row in audit))
        self.assertEqual([r["medical_specialty"] for r in records], ["Surgery", "Neurology"])


if __name__ == "__main__":
    unittest.main()
