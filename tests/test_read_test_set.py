"""Verificações da leitura do teste, de que depende o alinhamento do results.txt."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

from data_preparation import DatasetFormatError
from read_test_set import EXPECTED_TEST_CASES, TEST_COLUMNS, read_test_cases, write_results


def csv_text(rows, newline="\r\n", quoting=csv.QUOTE_MINIMAL):
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=";", lineterminator=newline, quoting=quoting).writerows(rows)
    return buffer.getvalue()


class ReadTestCasesTests(unittest.TestCase):
    def read(self, text, expected_cases=None):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test_no_labels.csv"
            source.write_bytes(text.encode("utf-8"))
            before = source.read_bytes()
            cases = read_test_cases(source, expected_cases=expected_cases)
            self.assertEqual(source.read_bytes(), before)
            return cases

    def test_one_case_per_record_with_quotes_preserved(self):
        rows = [[" d1 ", "n1", 'Texto com "aspas"; e separador', "k1"], ["d2", "n2", "t2", ""]]
        cases = self.read(csv_text(rows).removesuffix("\r\n"))
        self.assertEqual([[c[f] for f in TEST_COLUMNS] for c in cases], rows)
        self.assertEqual([c["case_id"] for c in cases], [1, 2])
        self.assertEqual([c["repair_actions"] for c in cases], [[], []])

    def test_valid_multiline_fields_are_preserved_without_label_heuristics(self):
        description = "Descrição\r\nSurgery;texto;que;continua"
        transcription = "Primeira linha\nSegunda linha"
        rows = [[description, "nome", transcription, ""], ["só descrição", "", "", ""]]
        cases = self.read(csv_text(rows))
        self.assertEqual([[c[f] for f in TEST_COLUMNS] for c in cases], rows)
        self.assertEqual((cases[0]["source_line_start"], cases[0]["source_line_end"]), (1, 3))
        self.assertEqual((cases[1]["source_line_start"], cases[1]["source_line_end"]), (4, 4))
        self.assertTrue(all(not case["repair_actions"] and not case["is_partial"] for case in cases))

    def test_optional_quoting_newlines_bom_and_final_newline_are_accepted(self):
        rows = [["descrição", "nome", "texto", ""], ["outro", "", "", ""]]
        for newline in ("\n", "\r\n"):
            for quoting in (csv.QUOTE_MINIMAL, csv.QUOTE_ALL):
                for bom in ("", "\ufeff"):
                    with self.subTest(newline=newline, quoting=quoting, bom=bool(bom)):
                        cases = self.read(bom + csv_text(rows, newline, quoting).removesuffix(newline))
                        self.assertEqual([[c[f] for f in TEST_COLUMNS] for c in cases], rows)

    def test_empty_cells_and_description_only_rows_are_not_discarded(self):
        rows = [["", "", "", ""], ["só descrição", "", "", ""], ["", "nome", "texto", ""]]
        cases = self.read(csv_text(rows), expected_cases=3)
        self.assertEqual([[c[f] for f in TEST_COLUMNS] for c in cases], rows)
        self.assertEqual([c["case_id"] for c in cases], [1, 2, 3])

    def test_header_is_rejected(self):
        for header in (TEST_COLUMNS, TEST_COLUMNS[::-1], [f" {name.upper()} " for name in TEST_COLUMNS]):
            with self.subTest(header=header), self.assertRaisesRegex(DatasetFormatError, "cabeçalho"):
                self.read(csv_text([header, ["d", "n", "t", "k"]]))

    def test_repeated_header_is_rejected_instead_of_becoming_a_case(self):
        with self.assertRaisesRegex(DatasetFormatError, "cabeçalho"):
            self.read(csv_text([["d", "n", "t", "k"], TEST_COLUMNS]))

    def test_wrong_field_count_is_rejected(self):
        for row in (["d", "n", "t"], ["Surgery", "d", "n", "t", "k"], []):
            with self.subTest(row=row), self.assertRaisesRegex(DatasetFormatError, "campos em vez de quatro"):
                self.read(csv_text([row]))

    def test_blank_physical_record_is_rejected_instead_of_silently_skipped(self):
        with self.assertRaises(DatasetFormatError):
            self.read("d;n;t;k\n\nd2;n2;t2;k2\n")

    def test_malformed_quoting_is_rejected(self):
        for text in ('"unfinished;n;t;k\n', '"d"extra;n;t;k\n'):
            with self.subTest(text=text), self.assertRaises(DatasetFormatError):
                self.read(text)

    def test_non_utf8_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.csv"
            source.write_bytes(b"descri\xe7ao;n;t;k\n")
            with self.assertRaisesRegex(DatasetFormatError, "UTF-8"):
                read_test_cases(source, expected_cases=None)

    def test_empty_file_is_rejected(self):
        with self.assertRaisesRegex(DatasetFormatError, "não contém registos"):
            self.read("")

    def test_explicit_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(DatasetFormatError, "2 casos em vez de 3"):
            self.read(csv_text([["d", "n", "t", "k"], ["d2", "n2", "t2", "k2"]]), expected_cases=3)

    def test_supplied_clean_snapshot_has_one_preserved_record_per_case(self):
        source = Path(__file__).resolve().parents[1] / "test_no_labels.csv"
        original = source.read_bytes()
        with source.open(encoding="utf-8-sig", newline="") as stream:
            expected_rows = list(csv.reader(stream, delimiter=";", strict=True))
        cases = read_test_cases(source)
        self.assertEqual(len(cases), EXPECTED_TEST_CASES)
        self.assertEqual(len(cases), len(expected_rows))
        self.assertEqual([c["case_id"] for c in cases], list(range(1, EXPECTED_TEST_CASES + 1)))
        self.assertTrue([[c[f] for f in TEST_COLUMNS] for c in cases] == expected_rows)
        self.assertEqual(source.read_bytes(), original)


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

    def test_invalid_predictions_leave_an_existing_result_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.txt"
            path.write_bytes(b"existing result\n")
            for predictions, cases in ((["Surgery"], [{}, {}]), (["Surgery\nNeurology"], [{}])):
                with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                    write_results(path, predictions, cases)
                self.assertEqual(path.read_bytes(), b"existing result\n")


if __name__ == "__main__":
    unittest.main()
