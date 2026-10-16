"""Train the final SVC with section augmentation and generate results.txt."""

import pandas as pd
from threadpoolctl import threadpool_limits

from augmentation import fit_evaluated_model
from config import RESULTS_FILE, SEED, TEST_FILE, TRAIN_FILE
from data_preparation import repair_training_csv
from duplicate_exclusion import fit_exclusion_index, score_model
from read_test_set import read_test_cases, write_results


def main():
    records, _ = repair_training_csv(TRAIN_FILE)
    train = pd.DataFrame(records)
    with threadpool_limits(limits=1):
        model, _ = fit_evaluated_model(train, "SVC_AUG", seed=SEED, fold=0)
    exclusion_index = fit_exclusion_index(train)

    cases = read_test_cases(TEST_FILE)
    test = pd.DataFrame(cases)
    scores, classes = score_model(model, test, "SVC_AUG")
    predictions = exclusion_index.predict(scores, classes, test).predictions
    write_results(RESULTS_FILE, [str(label) for label in predictions], cases)
    print(f"Wrote {len(predictions)} predictions to {RESULTS_FILE.name}.")


if __name__ == "__main__":
    main()
