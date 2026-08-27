"""
LLM-based ticket triage engine (LIVE Claude + offline stub).

For each ticket the engine returns a STRUCTURED JSON object:

    - intent        (one of the 77 BANKING77 intents)
    - urgency       (low | medium | high | critical)
    - sentiment     (negative | neutral | positive)
    - confidence    (0-1, the model's self-reported certainty in the intent)
    - entities      (amounts, card refs, dates, ... ; {} if none)
    - draft_reply   (a suggested first response the agent can edit and send)

Only `intent` has a ground-truth label in BANKING77, so only intent is scored
in the head-to-head. Urgency, sentiment, entities and the draft are produced by
the LLM but shown as unlabelled capabilities — the dataset gives us nothing to
grade them against, and inventing a grade would be dishonest.

Two execution modes
-------------------
1. LIVE mode  -- when ANTHROPIC_API_KEY is set, `classify_ticket` calls the real
   Claude Messages API with the engineered prompt below and parses the JSON.
   This is the only mode that produces a real LLM number for the benchmark.

2. OFFLINE mode -- when no key is set, the engine returns a schema-valid stub so
   the rest of the pipeline (enrichment shape, routing demo) still runs on any
   machine. IMPORTANT: on 77-way real intents an offline heuristic is not a
   language model and is NOT representative of Claude, so the offline stub
   deliberately returns intent=None and is never scored as "the LLM". Run LIVE
   to populate the head-to-head.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache

# Urgency / sentiment vocabularies the LLM must choose from.
URGENCY_LEVELS = ["low", "medium", "high", "critical"]
SENTIMENTS = ["negative", "neutral", "positive"]

DEFAULT_MODEL = "claude-sonnet-4-20250514"


@lru_cache(maxsize=1)
def _intent_list() -> list[str]:
    """The 77 real intent labels, loaded from the training split."""
    from src.data_loader import load_split, intent_labels
    return intent_labels(load_split("train"))


def build_system_prompt() -> str:
    """The exact system prompt used in LIVE mode, listing the real intents."""
    intents = ", ".join(_intent_list())
    return f"""You are an intent-triage assistant for a retail bank's customer support queue.
You read one inbound customer message and return a STRICT JSON object a routing
system will consume. Never add commentary outside the JSON.

Rules:
- intent MUST be exactly one of these 77 labels:
{intents}
- urgency: one of low | medium | high | critical
    critical = money actively at risk or account compromised
    high = strongly time-sensitive or blocked from using the account
    medium = normal request expecting timely follow-up
    low = casual question, no time pressure
- sentiment: one of negative | neutral | positive
- confidence: your certainty in the intent, 0.0 to 1.0
- entities: pull out amounts, card/account references, or dates if present (else {{}})
- draft_reply: a concise, empathetic 2-3 sentence first response the agent can edit and send

Return ONLY the JSON object."""


USER_PROMPT_TEMPLATE = """Triage this customer message.

Message: {text}

Return JSON with keys: intent, urgency, sentiment, confidence, entities, draft_reply."""


# ---------------------------------------------------------------------------
# LIVE mode: real Claude API call
# ---------------------------------------------------------------------------

def _classify_live(text: str, model: str) -> dict:
    import anthropic  # imported lazily so offline mode needs no dependency

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    out = json.loads(raw)
    out.setdefault("entities", {})
    out.setdefault("confidence", 0.8)
    out["offline_stub"] = False
    return out


# ---------------------------------------------------------------------------
# OFFLINE mode: schema-valid, explicitly NON-representative stub
# ---------------------------------------------------------------------------

_NEG = ["stolen", "lost", "compromised", "unauthorised", "unauthorized", "wrong",
        "declined", "not recognised", "not recognized", "failed", "stuck", "still",
        "charged twice", "can't", "cannot", "won't", "unable", "urgent", "asap"]
_POS = ["thanks", "thank you", "great", "love", "please", "curious", "interested"]
_URGENT = ["stolen", "compromised", "unauthorised", "unauthorized", "fraud",
           "asap", "urgent", "immediately", "right now", "declined"]


def _heuristic_urgency(t: str) -> str:
    tl = t.lower()
    if any(w in tl for w in _URGENT):
        return "high"
    return "medium"


def _heuristic_sentiment(t: str) -> str:
    tl = t.lower()
    neg = sum(w in tl for w in _NEG)
    pos = sum(w in tl for w in _POS)
    if neg > pos and neg:
        return "negative"
    if pos > neg and pos:
        return "positive"
    return "neutral"


def _extract_entities(t: str) -> dict:
    ent: dict = {}
    m = re.search(r"[£$€]\s?(\d+(?:\.\d+)?)", t)
    if m:
        ent["amount"] = m.group(0)
    m = re.search(r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", t)
    if m:
        ent["date"] = m.group(1)
    return ent


def _classify_offline(text: str) -> dict:
    """Schema-valid stub. intent is intentionally None so it is never scored."""
    return {
        "intent": None,                      # not a language model — do not score
        "urgency": _heuristic_urgency(text),
        "sentiment": _heuristic_sentiment(text),
        "confidence": 0.0,
        "entities": _extract_entities(text),
        "draft_reply": ("Thanks for getting in touch — I can see what you're asking "
                        "about and I'm looking into it now. I'll follow up shortly "
                        "with the details and next steps."),
        "offline_stub": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_live_mode() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def classify_ticket(text: str, model: str = DEFAULT_MODEL, force_offline: bool = False) -> dict:
    """Classify a single ticket into the structured triage object."""
    if not force_offline and is_live_mode():
        try:
            return _classify_live(text, model)
        except Exception:  # network/parse failure -> schema-valid fallback
            return _classify_offline(text)
    return _classify_offline(text)


def classify_batch(texts, force_offline: bool = False, sleep: float = 0.0) -> list[dict]:
    """Classify an iterable of ticket texts."""
    out = []
    for t in texts:
        out.append(classify_ticket(t, force_offline=force_offline))
        if sleep:
            time.sleep(sleep)
    return out


def estimate_live_cost(n_tickets: int,
                       in_tokens_per_call: int = 750,
                       out_tokens_per_call: int = 120,
                       in_price_per_mtok: float = 3.0,
                       out_price_per_mtok: float = 15.0) -> dict:
    """Rough $ estimate for running the LIVE head-to-head over n tickets.

    Defaults assume Claude Sonnet-class pricing and the ~77-label prompt above.
    Adjust the token/price figures for the model you actually run.
    """
    cost = (n_tickets * in_tokens_per_call / 1e6) * in_price_per_mtok \
         + (n_tickets * out_tokens_per_call / 1e6) * out_price_per_mtok
    return {"n_tickets": n_tickets, "est_usd": round(cost, 2)}
