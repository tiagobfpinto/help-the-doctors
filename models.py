"""Model definitions.

Each model is a scikit-learn Pipeline: text -> features -> classifier.
Keeping the feature extraction inside the Pipeline matters: during
cross-validation the vocabulary and IDF are refitted on each training fold
only, so the validation texts never influence the features.
"""

import re
from collections.abc import Sequence
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default word pattern
_STEMMER = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _STEMMER.stem(word)


def stemmed_tokens(text: str) -> list[str]:
    """Split text into words and reduce each to its stem ("fractures" -> "fractur")."""
    return [_stem(word) for word in TOKEN_PATTERN.findall(text)]


def naive_bayes(alpha: float = 1.0, fit_prior: bool = True, stem: bool = False) -> Pipeline:
    """The professors' baseline: TF-IDF of lowercased words + Multinomial Naive Bayes.

    alpha: smoothing added to every word count. The default 1.0 is strong
        compared with TF-IDF values, which are mostly below 0.5.
    fit_prior: start from the class frequencies (True) or treat every class
        as equally likely (False).
    stem: reduce words to their stems with the Porter stemmer. The project
        baseline uses stemming, so naive_bayes(stem=True) reproduces it.
    """
    tokenizer = {"tokenizer": stemmed_tokens, "token_pattern": None} if stem else {}
    return Pipeline([
        ("counts", CountVectorizer(lowercase=True, **tokenizer)),
        ("tfidf", TfidfTransformer()),
        ("classifier", MultinomialNB(alpha=alpha, fit_prior=fit_prior)),
    ])


def logistic_regression(
    fields: Sequence[str] = ("description", "sample_name", "transcription"),
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    random_state: int = 0,
) -> Pipeline:
    """Separate TF-IDF per field, followed by a balanced linear classifier.

    Input is a DataFrame with the selected text columns, with missing values
    represented as empty strings. A scalar column selector gives each
    vectorizer a one-dimensional sequence of texts. Short fields keep their
    own normalized features instead of being diluted by long transcripts.
    Vocabulary and IDF are learned inside each training fold.
    """
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Choose at least one text field, without repetitions.")
    allowed = {"description", "sample_name", "transcription"}
    if unknown := set(fields) - allowed:
        raise ValueError(f"Unsupported text fields (keywords reveal the label): {sorted(unknown)}")

    features = ColumnTransformer([
        (field, TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        ), field)
        for field in fields
    ], remainder="drop")
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])
