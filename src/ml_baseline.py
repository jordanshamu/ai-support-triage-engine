"""
Traditional ML baseline: TF-IDF + Logistic Regression.

This is the "control" against which the LLM triage engine is benchmarked. It
represents the classic pre-LLM approach to text classification: a supervised
model trained on labelled historical tickets.

The comparison answers a question every analytics team faces in 2026:
    "When is a zero-shot LLM worth the per-call cost and latency versus a
     cheap, fast, trained classifier?"

The baseline only predicts `intent`. Urgency and sentiment, and the auto-drafted
reply, are capabilities the LLM provides out-of-the-box with zero training data
-- an advantage this dataset cannot score (it has no such labels), so it is shown
qualitatively rather than measured.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline() -> Pipeline:
    """Construct the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=4.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def make_text(df):
    """Return the text field for vectorisation.

    BANKING77 ships one text column per ticket (no subject/message split), so
    this simply hands the cleaned customer message to the vectoriser.
    """
    from src.data_loader import TEXT_COL
    return df[TEXT_COL].fillna("").tolist()
