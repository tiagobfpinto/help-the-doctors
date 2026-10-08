"""Verificações da leitura do teste, de que depende o alinhamento do results.txt."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

from data_preparation import DatasetFormatError
from read_test_set import TEST_COLUMNS, read_test_cases, write_results


def csv_text(rows):
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=";", lineterminator="\r\n").writerows(rows)
    return buffer.getvalue()


class ReadTestCasesTests(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test_no_labels.csv"
            source.write_bytes(text.encode("utf-8"))
            return read_test_cases(source, expected_cases=None)

    def test_one_case_per_record_with_quotes_preserved(self):
        rows = [["d1", "n1", 'Texto com "aspas"; e separador', "k1"], ["d2", "n2", "t2", ""]]
        cases = self.read(csv_text(rows).removesuffix("\r\n"))
        self.assertEqual([[c[f] for f in TEST_COLUMNS] for c in cases], rows)
        self.assertEqual([c["repair_actions"] for c in cases], [[], []])

    def test_embedded_lines_are_not_cases_and_continuation_is_kept(self):
        merged = " Descrição própria\r\nSurgery;d;n;t;k\r\nNeurology;d;n;t parcial"
        cases = self.read(csv_text([
            ["d0", "n0", "t0", "k0"], [merged, "", "", ""],
            ["neurology, keywords, ", "", "", ""], ["d3", "n3", "t3", "k3"],
        ]))
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[1]["description"], " Descrição própria")
        self.assertEqual(cases[1]["embedded_specialties"], "Surgery;Neurology")
        self.assertEqual((cases[1]["source_line_start"], cases[1]["source_line_end"]), (2, 4))
        self.assertEqual(cases[2]["repair_actions"], ["flag_continuation_row"])
        self.assertEqual(cases[2]["description"], "neurology, keywords, ")
        self.assertEqual(cases[3]["source_line_start"], 6)

    def test_description_only_case_without_merge_is_not_flagged(self):
        cases = self.read(csv_text([["só descrição", "", "", ""]]))
        self.assertEqual(cases[0]["repair_actions"], [])

    def test_embedded_line_without_specialty_is_rejected(self):
        with self.assertRaises(DatasetFormatError):
            self.read(csv_text([["Descrição\r\ntexto solto;d;n", "", "", ""]]))

    def test_header_is_rejected(self):
        with self.assertRaises(DatasetFormatError):
            self.read(csv_text([TEST_COLUMNS, ["d", "n", "t", "k"]]))

    def test_wrong_field_count_is_rejected(self):
        with self.assertRaises(DatasetFormatError):
            self.read(csv_text([["Surgery", "d", "n", "t", "k"]]))

    def test_non_canonical_quoting_is_rejected(self):
        with self.assertRaises(DatasetFormatError):
            self.read('"d";n;t;k\r\n')


class WriteResultsTests(unittest.TestCase):
    def test_one_label_per_case_without_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.txt"
            write_results(path, ["Surgery", "Neurology"], [{}, {}])
            self.assertEqual(path.read_bytes(), b"Surgery\nNeurology\n")

    def test_count_mismatch_and_unknown_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.txt"
            with self.assertRaises(ValueError):
                write_results(path, ["Surgery"], [{}, {}])
            with self.assertRaises(ValueError):
                write_results(path, ["Cardiology"], [{}])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
