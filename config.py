"""Paths and settings shared by every script.

Change experiment settings here, not inside the scripts.
"""

from pathlib import Path

# Paths are relative to this file, so the project works on any machine.
ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "train.csv"
TEST_FILE = ROOT / "test_no_labels.csv"
RESULTS_FILE = ROOT / "results.txt"
FIGURES_DIR = ROOT / "outputs" / "figures"

LABEL = "medical_specialty"

# Default fields for field-aware linear models. Naive Bayes explicitly uses
# description. Keywords are excluded because they name the target.
# Final inference uses SVC_AUG from augmentation.py.
TEXT_FIELDS = ("sample_name", "transcription")
CONTROLLED_OUTPUT_DIR = ROOT / "outputs" / "augmentation_controlada"
FINAL_EVALUATION_DIR = ROOT / "outputs" / "avaliacao_recombinacao"
FINAL_OUTPUT_DIR = ROOT / "outputs" / "final"

# Cross-validation
N_FOLDS = 5
SEED = 0

# Pass criteria of the automatic evaluation
MIN_ACCURACY = 0.49
MIN_CLASS_F1 = 0.25
