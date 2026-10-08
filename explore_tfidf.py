"""Look at the TF and TF-IDF matrices of the training set as heatmaps.

Exploration only: no model is trained here. Run from the project folder:
    .venv\\Scripts\\python.exe explore_tfidf.py

The figures are shown and saved in outputs/figures/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from config import FIGURES_DIR, LABEL, SEED
from data import load_train

FIELD = "description"
DOCS_PER_CLASS = 2   # rows of each heatmap, per specialty
TF_WORDS = 30        # columns of the TF heatmap: most frequent words in the corpus
TFIDF_WORDS_PER_DOC = 2  # columns of the TF-IDF heatmap: top words of each row

# Chart colours: one hue, light to dark, so a zero fades into the background.
SURFACE, INK, MUTED_INK = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list(
    "weights", [SURFACE, "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])


def main() -> None:
    train = load_train()

    # TF: how many times each word appears in each document.
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(train[FIELD])
    words = vectorizer.get_feature_names_out()

    # TF-IDF: counts multiplied by idf = ln((1 + n_docs) / (1 + docs_with_word)) + 1,
    # then each row scaled to length 1. Words found everywhere get small weights.
    idf_step = TfidfTransformer()
    tfidf = idf_step.fit_transform(tf)

    idf = pd.Series(idf_step.idf_, index=words)
    print(f"Matrix: {tf.shape[0]} documents x {tf.shape[1]} words")
    print("Lowest IDF (most common words):", idf.nsmallest(6).round(2).to_dict())

    # The full matrices are too big to read, so plot a few documents of each
    # specialty. The CSV is grouped by specialty: the first rows share one label.
    sample = train.groupby(LABEL).sample(DOCS_PER_CLASS, random_state=SEED)
    row_names = [f"{label}  (row {i})" for i, label in sample[LABEL].items()]

    frequent = most_frequent_words(tf, TF_WORDS)
    plot_heatmap(
        tf[sample.index][:, frequent].toarray(), row_names, words[frequent],
        title=f"TF (counts) in {FIELD}: {DOCS_PER_CLASS} documents per specialty, "
              f"{TF_WORDS} most frequent words",
        scale_label="Count", file_name="tf_heatmap.png", value_format="d",
    )

    sample_tfidf = tfidf[sample.index].toarray()
    top = top_words_per_row(sample_tfidf, TFIDF_WORDS_PER_DOC)
    plot_heatmap(
        sample_tfidf[:, top], row_names, words[top],
        title=f"TF-IDF in {FIELD}: same documents, top {TFIDF_WORDS_PER_DOC} words of each",
        scale_label="TF-IDF weight", file_name="tfidf_heatmap.png", value_format=".2f",
    )
    plt.show()


def most_frequent_words(tf, n: int) -> np.ndarray:
    """Column indices of the n words with the highest total count."""
    totals = np.asarray(tf.sum(axis=0)).ravel()
    return totals.argsort()[::-1][:n]


def top_words_per_row(weights: np.ndarray, n: int) -> list[int]:
    """Column indices of each row's n highest weights, without repeats, in row order."""
    columns: list[int] = []
    for row in weights:
        for j in row.argsort()[::-1][:n]:
            if row[j] and j not in columns:
                columns.append(j)
    return columns


def plot_heatmap(values: np.ndarray, row_names, column_names, *, title: str,
                 scale_label: str, file_name: str, value_format: str) -> None:
    """Draw one heatmap with the value written in every non-zero cell, and save it."""
    n_rows, n_cols = values.shape
    fig, ax = plt.subplots(figsize=(0.38 * n_cols + 4, 0.34 * n_rows + 2.2), facecolor=SURFACE)
    mesh = ax.pcolormesh(values, cmap=BLUES, vmin=0, edgecolors=SURFACE, linewidth=1.5)

    for (r, c), value in np.ndenumerate(values):
        if value:
            text_colour = "#ffffff" if value > values.max() / 2 else INK
            ax.text(c + 0.5, r + 0.5, format(value, value_format),
                    ha="center", va="center", fontsize=7, color=text_colour)

    ax.set_xticks(np.arange(n_cols) + 0.5, column_names, rotation=60, ha="right", color=MUTED_INK)
    ax.set_yticks(np.arange(n_rows) + 0.5, row_names, color=MUTED_INK)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK)

    integer_ticks = range(int(values.max()) + 1) if value_format == "d" else None
    bar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02, ticks=integer_ticks)
    bar.set_label(scale_label, color=MUTED_INK)
    bar.outline.set_visible(False)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / file_name, dpi=150)


if __name__ == "__main__":
    main()
