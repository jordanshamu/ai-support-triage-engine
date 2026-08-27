"""
Business impact model.

Translates classifier behaviour into money. Two pieces:

1. Confidence gating (human-in-the-loop design)
   Tickets the model is confident about AND that are not high-stakes are
   auto-handled (routed + draft reply sent for agent one-click approval).
   Everything else is escalated to a human. This is the core safety mechanism
   of any production automation: never let a low-confidence or high-risk
   decision go out unsupervised.

2. ROI model
   Given monthly ticket volume, agent handling time, loaded labour cost, and
   the model's automation/accuracy behaviour, compute annual savings, cost of
   running the LLM, and net ROI.
"""

from __future__ import annotations

# Intents that are never fully auto-resolved regardless of confidence, because
# the downside of a wrong call is high (money at risk, security/fraud, or a
# customer we could lose). Curated from the real BANKING77 labels.
from src.data_loader import HIGH_STAKES  # noqa: F401  (re-exported for callers)


def gate_decision(pred_intent: str, confidence: float,
                  conf_threshold: float = 0.75) -> str:
    """Return 'auto' or 'human' for a single predicted ticket.

    auto  -> routed automatically + draft reply queued for one-click send
    human -> escalated to a support agent for full handling

    High-stakes intents always go to a human. Everything else is auto-handled
    only when the classifier is confident enough (>= conf_threshold).
    """
    if pred_intent in HIGH_STAKES:
        return "human"
    if confidence >= conf_threshold:
        return "auto"
    return "human"


def automation_summary(preds: list[dict], conf_threshold: float = 0.75) -> dict:
    """Compute auto vs human split across a batch of predictions."""
    decisions = [
        gate_decision(p["intent"], float(p.get("confidence", 0.0)), conf_threshold)
        for p in preds
    ]
    n = len(decisions)
    auto = sum(1 for d in decisions if d == "auto")
    return {
        "total_tickets": n,
        "auto_handled": auto,
        "human_handled": n - auto,
        "automation_rate": round(auto / n, 4) if n else 0.0,
        "conf_threshold": conf_threshold,
        "decisions": decisions,
    }


def roi_model(
    automation_rate: float,
    auto_accuracy: float,
    monthly_volume: int = 12000,
    minutes_per_ticket_manual: float = 9.0,
    minutes_per_ticket_assisted: float = 2.5,
    loaded_hourly_cost: float = 32.0,
    llm_cost_per_ticket: float = 0.012,
    rework_minutes_per_error: float = 12.0,
) -> dict:
    """Compute annual ROI of deploying the triage engine.

    Parameters
    ----------
    automation_rate          : share of tickets auto-handled (from gating)
    auto_accuracy            : routing accuracy on auto-handled tickets
    monthly_volume           : inbound tickets per month
    minutes_per_ticket_manual: handling time without the engine
    minutes_per_ticket_assisted: handling time for auto-handled (review + send)
    loaded_hourly_cost       : fully-loaded agent cost per hour
    llm_cost_per_ticket      : API cost to classify one ticket
    rework_minutes_per_error : extra time to fix a misrouted ticket
    """
    annual_volume = monthly_volume * 12
    cost_per_minute = loaded_hourly_cost / 60.0

    # Baseline: every ticket handled manually.
    baseline_cost = annual_volume * minutes_per_ticket_manual * cost_per_minute

    auto_tickets = annual_volume * automation_rate
    human_tickets = annual_volume - auto_tickets

    # Auto tickets: agent only reviews + clicks send.
    auto_handling_cost = auto_tickets * minutes_per_ticket_assisted * cost_per_minute
    # Human tickets: still manual, but pre-triaged so a small time saving applies.
    human_handling_cost = human_tickets * (minutes_per_ticket_manual * 0.85) * cost_per_minute

    # Misrouting rework on the automated stream.
    misrouted = auto_tickets * (1 - auto_accuracy)
    rework_cost = misrouted * rework_minutes_per_error * cost_per_minute

    # LLM inference cost on every ticket.
    llm_cost = annual_volume * llm_cost_per_ticket

    engine_total_cost = auto_handling_cost + human_handling_cost + rework_cost + llm_cost
    net_savings = baseline_cost - engine_total_cost
    roi_multiple = (net_savings / (llm_cost + rework_cost)) if (llm_cost + rework_cost) else 0.0

    # Hours of agent capacity freed up.
    hours_saved = (baseline_cost - (auto_handling_cost + human_handling_cost)) / loaded_hourly_cost

    return {
        "annual_volume": int(annual_volume),
        "baseline_annual_cost": round(baseline_cost, 2),
        "engine_annual_cost": round(engine_total_cost, 2),
        "net_annual_savings": round(net_savings, 2),
        "savings_pct": round(net_savings / baseline_cost, 4) if baseline_cost else 0.0,
        "roi_multiple": round(roi_multiple, 2),
        "agent_hours_freed": round(hours_saved, 0),
        "llm_annual_cost": round(llm_cost, 2),
        "rework_annual_cost": round(rework_cost, 2),
        "auto_tickets": int(auto_tickets),
        "human_tickets": int(human_tickets),
        "assumptions": {
            "monthly_volume": monthly_volume,
            "minutes_per_ticket_manual": minutes_per_ticket_manual,
            "minutes_per_ticket_assisted": minutes_per_ticket_assisted,
            "loaded_hourly_cost": loaded_hourly_cost,
            "llm_cost_per_ticket": llm_cost_per_ticket,
            "rework_minutes_per_error": rework_minutes_per_error,
        },
    }
