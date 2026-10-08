"""Load the train and test sets as DataFrames.

Both are read from the original files through the same repair code that
data_preparation.py and read_test_set.py use, so there is one source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from config import TEST_FILE, TRAIN_FILE
from data_preparation import COLUMNS, repair_training_csv
from read_test_set import TEST_COLUMNS, read_test_cases


def load_train() -> pd.DataFrame:
    """The 2,669 repaired training cases, with the label and four text fields."""
    records, _ = repair_training_csv(TRAIN_FILE)
    return pd.DataFrame(records)[COLUMNS]


def load_test() -> pd.DataFrame:
    """The 409 test cases, in the order of results.txt. Never use them for training."""
    return pd.DataFrame(read_test_cases(TEST_FILE))[TEST_COLUMNS]


def text_input(frame: pd.DataFrame, fields: Sequence[str]) -> pd.Series:
    """Join the chosen text fields of each case into one string."""
    return frame[list(fields)].agg(" ".join, axis=1)
