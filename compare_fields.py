"""Does the model improve with more input fields?

Same folds, same seed and same models in every row: only the text fields change.
Run from the project folder:
    .venv\\Scripts\\python.exe compare_fields.py
"""

from config import LABEL, MIN_ACCURACY, MIN_CLASS_F1
from data import load_train, text_input
from evaluation import out_of_fold_predictions, summary
from models import naive_bayes

FIELD_SETS = [
    ("description",),
    ("sample_name",),
    ("transcription",),
    ("description", "sample_name"),
    ("description", "sample_name", "transcription"),
    # Reference only: keywords start with the specialty name, so this row leaks the label.
    ("description", "sample_name", "transcription", "keywords"),
]

MODELS = {
    "NB default": lambda: naive_bayes(),
    "NB alpha=0.03, equal priors": lambda: naive_bayes(alpha=0.03, fit_prior=False),
}


def main() -> None:
    train = load_train()
    labels = train[LABEL]

    print(f"{'Fields':<52} {'Model':<28} {'Accuracy':>8} {'Mean F1':>8} {'Min F1':>7}  Passes")
    for fields in FIELD_SETS:
        texts = text_input(train, fields)
        for model_name, build_model in MODELS.items():
            scores = summary(labels, out_of_fold_predictions(build_model(), texts, labels))
            passes = scores["accuracy"] > MIN_ACCURACY and scores["min_f1"] >= MIN_CLASS_F1
            print(f"{' + '.join(fields):<52} {model_name:<28} {scores['accuracy']:>8.1%} "
                  f"{scores['mean_f1']:>8.1%} {scores['min_f1']:>7.1%}  {'yes' if passes else 'no'}")


if __name__ == "__main__":
    main()
