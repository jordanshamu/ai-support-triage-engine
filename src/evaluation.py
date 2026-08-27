"""
Evaluation utilities.

Shared scoring helpers so both the LLM engine and the ML baseline are judged on
an identical footing: accuracy, macro-F1, weighted-F1, per-class
precision/recall, and confusion matrices.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def score(y_true, y_pred, labels) -> dict:
    """Return a compact metrics dict for a single classifier."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        lab: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4),
            "support": int(s[i]),
        }
        for i, lab in enumerate(labels)
    }
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
    }


def cm(y_true, y_pred, labels) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def text_report(y_true, y_pred, labels) -> str:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)
