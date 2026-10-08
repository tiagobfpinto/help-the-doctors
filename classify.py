"""Evaluate a model with cross-validation, then predict the test set.

Run from the project folder:
    .venv\\Scripts\\python.exe classify.py

To try another model or other text fields, change `build_model` below or
TEXT_FIELDS in config.py.
"""

from collections import Counter

from sklearn.pipeline import Pipeline

from config import LABEL, RESULTS_FILE, TEXT_FIELDS
from data import load_test, load_train, text_input
from evaluation import out_of_fold_predictions, report
from models import naive_bayes
from read_test_set import write_results


def build_model() -> Pipeline:
    """The model this script evaluates and uses for results.txt."""
    return naive_bayes()


def main() -> None:
    train = load_train()
    test = load_test()
    train_texts = text_input(train, TEXT_FIELDS)
    train_labels = train[LABEL]
    print(f"Train: {len(train)} cases | Test: {len(test)} cases | Fields: {', '.join(TEXT_FIELDS)}\n")

    # 1. Estimate how good the model is, using only the training data.
    passed = report(train_labels, out_of_fold_predictions(build_model(), train_texts, train_labels))
    if not passed:
        print("Warning: this model does not meet the pass criteria.")

    # 2. Train on every training case and predict the test cases, in file order.
    model = build_model().fit(train_texts, train_labels)
    predictions = [str(label) for label in model.predict(text_input(test, TEXT_FIELDS))]
    write_results(RESULTS_FILE, predictions, test)

    print(f"\nWrote {len(predictions)} predictions to {RESULTS_FILE}")
    print("Predicted on test:", dict(Counter(predictions).most_common()))


if __name__ == "__main__":
    main()
