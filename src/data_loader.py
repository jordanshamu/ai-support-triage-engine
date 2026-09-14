"""
Real dataset loader — BANKING77.

This project was rebuilt on a genuinely real, public, labelled dataset so the
benchmark rests on data the analyst did not generate.

Dataset: BANKING77 (Casanueva et al., "Efficient Intent Detection with Dual
Sentence Encoders", NLP4ConvAI @ ACL 2020). 13,083 real online-banking customer
service queries, each labelled with one of 77 fine-grained intents. Released by
PolyAI under CC-BY-4.0.

    Source: https://github.com/PolyAI-LDN/task-specific-datasets
    Split : train 10,003 rows · test 3,080 rows (test is balanced, 40/intent)
    Fields: text (the customer message) · category (the intent label)

Why this dataset
----------------
It is real customer-support text with a real category label, which is exactly
what the trained baseline needs to learn something and what a fair LLM-vs-ML
benchmark needs to score against. Its 77 fine-grained intents also make it hard
in an honest way: several intents are near-synonyms (e.g. `verify_my_identity`
vs `why_verify_identity` vs `unable_to_verify_identity`), so the errors are
concentrated and interpretable rather than random.

What it does NOT contain
------------------------
There are no urgency, sentiment, or entity labels. So the *scored* head-to-head
is intent classification only. The LLM engine still produces urgency, sentiment,
entities, and a draft reply, but those are shown as unlabelled capabilities, not
scored against a ground truth this dataset does not have. That honesty is the
point of the rebuild.
"""

from __future__ import annotations

import os
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# Source + local cache
# ---------------------------------------------------------------------------

_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
_URLS = {"train": f"{_BASE}/train.csv", "test": f"{_BASE}/test.csv"}

DATASET_NAME = "BANKING77"
DATASET_SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets"
DATASET_LICENSE = "CC-BY-4.0"

# The single text field and the label field, after we normalise column names.
TEXT_COL = "text"
LABEL_COL = "intent"

# Intents whose downside — money at risk, security, fraud, or a customer we
# could lose — is high enough that a prediction of one is never auto-resolved,
# whatever the model's confidence. Curated from the real 77 labels; the routing
# layer sends these to a human every time. The gate reads the *predicted*
# intent, so it cannot catch a high-stakes ticket the model confidently files as
# routine — see business_impact.gate_decision and high_stakes_leakage.
HIGH_STAKES = frozenset({
    "compromised_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "card_swallowed",
    "cash_withdrawal_not_recognised",
    "card_payment_not_recognised",
    "direct_debit_payment_not_recognised",
    "transaction_charged_twice",
    "wrong_amount_of_cash_received",
    "unable_to_verify_identity",
    "transfer_not_received_by_recipient",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "reverted_card_payment?",   # yes — this real label ships with a trailing '?'
})


def _cache_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "data", "raw")
    os.makedirs(path, exist_ok=True)
    return path


def _download(split: str) -> str:
    """Fetch a split to the local cache if not already present; return its path."""
    dest = os.path.join(_cache_dir(), f"banking77_{split}.csv")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(_URLS[split], dest)
    return dest


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the (deliberately light) cleaning the real data actually needs.

    Banking77 is clean: no nulls, no exact duplicates, no train/test text
    overlap, and every text maps to exactly one intent. The only real issues
    are 9 rows with stray leading/trailing whitespace and the fact that text
    is short. So we strip whitespace, drop anything empty after stripping, and
    rename columns to a stable schema. We do NOT invent fields the data lacks.
    """
    df = df.rename(columns={"text": TEXT_COL, "category": LABEL_COL}).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].reset_index(drop=True)
    return df


def load_split(split: str) -> pd.DataFrame:
    """Load and clean a single split ('train' or 'test')."""
    if split not in _URLS:
        raise ValueError(f"split must be one of {list(_URLS)}")
    return clean(pd.read_csv(_download(split)))


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean both splits. Returns (train_df, test_df)."""
    return load_split("train"), load_split("test")


def intent_labels(train_df: pd.DataFrame) -> list[str]:
    """The sorted list of intent labels present in the data (77 of them)."""
    return sorted(train_df[LABEL_COL].unique())
