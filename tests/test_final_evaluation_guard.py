"""No new fit or output write may precede frozen manual review."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_final_evaluation as runner


class EvaluationGuardTests(unittest.TestCase):
    def test_missing_plan_blocks_all_fits_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new_output"
            with patch.multiple(runner, ROOT=root, OUT=output), \
                 patch.object(runner, "fit_evaluated_model") as fit:
                with self.assertRaisesRegex(ValueError, "No frozen"):
                    runner.run()
                fit.assert_not_called()
                self.assertFalse(output.exists())

    def test_existing_evaluation_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "run_state.json"
            state.write_text('{"state":"complete"}', encoding="utf-8")
            with patch.object(runner, "OUT", root), \
                 patch.object(runner, "validate_decision_plan", return_value={}), \
                 patch.object(runner, "fit_evaluated_model") as fit:
                with self.assertRaises(FileExistsError):
                    runner.run()
                fit.assert_not_called()
                self.assertEqual(state.read_text(encoding="utf-8"), '{"state":"complete"}')


if __name__ == "__main__":
    unittest.main()
