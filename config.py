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

# Text columns the model reads, joined into one string per case.
# Options: "description", "sample_name", "transcription", "keywords".
# Warning: non-empty keywords always start with the specialty name (label leakage).
TEXT_FIELDS = ("description",)

# Cross-validation
N_FOLDS = 5
SEED = 0

# Pass criteria of the automatic evaluation
MIN_ACCURACY = 0.49
MIN_CLASS_F1 = 0.25
