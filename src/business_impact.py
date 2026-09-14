"""
Business impact model.

Translates classifier behaviour into money. Two pieces:

1. Confidence gating (human-in-the-loop design)
   Tickets the model is confident about AND that it does not predict as
   high-stakes are auto-handled (routed + draft reply sent for agent one-click
   approval). Everything else is escalated to a human.

   The limit of this design, stated plainly because it matters: the gate sees
   the *prediction*, not the truth. It stops low-confidence calls and tickets
   the model knows are risky, but a genuinely high-stakes ticket that the model
   confidently mislabels as routine passes straight through. `high_stakes_leakage`
   measures that residual against ground truth so the number is reported rather
   than assumed.

2. ROI model
   Given monthly ticket volume, agent handling time, loaded labour cost, and
   the model's automation/accuracy behaviour, compute annual savings, cost of
   running the LLM, and net ROI.
"""

from __future__ import annotations

# Intents whose downside is high enough (money at risk, security/fraud, or a
# customer we could lose) that a prediction of one is never auto-resolved,
# whatever the confidence. Curated from the real BANKING77 labels.
from src.data_loader import HIGH_STAKES  # noqa: F401  (re-exported for callers)


def gate_decision(pred_intent: str, confidence: float,
                  conf_threshold: float = 0.75) -> str:
    """Return 'auto' or 'human' for a single predicted ticket.

    auto  -> routed automatically + draft reply queued for one-click send
    human -> escalated to a support agent for full handling

    A ticket *predicted* as high-stakes always goes to a human, whatever the
    confidence. Everything else is auto-handled only when the classifier is
    confident enough (>= conf_threshold).

    Note what this does and does not guarantee. The gate is applied to
    `pred_intent`, so it cannot catch a high-stakes ticket the model confidently
    files as routine — see `high_stakes_leakage` for the measured residual.
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


def high_stakes_leakage(preds: list[dict], conf_threshold: float = 0.75,
                        max_examples: int = 5) -> dict:
    """Audit the gate against ground truth: how many real high-stakes tickets slip through?

    `gate_decision` can only act on the predicted intent, so this is the number
    that says what the guardrail is actually worth. Each pred needs a
    `true_intent` alongside the predicted `intent` and `confidence`.

    A non-zero leak is not a bug in the gate — it is the cost of routing on a
    prediction. Reporting it is what makes the automation rate honest.
    """
    true_hs = [p for p in preds if p["true_intent"] in HIGH_STAKES]
    leaked = [
        p for p in true_hs
        if gate_decision(p["intent"], float(p.get("confidence", 0.0)), conf_threshold) == "auto"
    ]
    n_all, n_hs = len(preds), len(true_hs)
    return {
        "conf_threshold": conf_threshold,
        "true_high_stakes_tickets": n_hs,
        "leaked_to_auto": len(leaked),
        "leak_rate_of_high_stakes": round(len(leaked) / n_hs, 4) if n_hs else 0.0,
        "leak_rate_of_all_tickets": round(len(leaked) / n_all, 4) if n_all else 0.0,
        "examples": [
            {"true_intent": p["true_intent"], "pred_intent": p["intent"],
             "confidence": round(float(p.get("confidence", 0.0)), 4)}
            for p in leaked[:max_examples]
        ],
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
    pretriage_time_saving: float = 0.15,
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
    pretriage_time_saving    : fraction of handling time saved on *escalated*
                               tickets, which still reach an agent already
                               classified and with a draft attached. This is an
                               assumption, not a measured effect, and a material
                               one — it is worth ~$59K of the modelled saving at
                               the default volume. Set it to 0.0 for the
                               conservative case.
    """
    annual_volume = monthly_volume * 12
    cost_per_minute = loaded_hourly_cost / 60.0

    # Baseline: every ticket handled manually.
    baseline_cost = annual_volume * minutes_per_ticket_manual * cost_per_minute

    auto_tickets = annual_volume * automation_rate
    human_tickets = annual_volume - auto_tickets

    # Auto tickets: agent only reviews + clicks send.
    auto_handling_cost = auto_tickets * minutes_per_ticket_assisted * cost_per_minute
    # Human tickets: still manual, but pre-triaged so an assumed time saving applies.
    human_handling_cost = (
        human_tickets * (minutes_per_ticket_manual * (1 - pretriage_time_saving)) * cost_per_minute
    )

    # Misrouting rework on the automated stream.
    misrouted = auto_tickets * (1 - auto_accuracy)
    rework_cost = misrouted * rework_minutes_per_error * cost_per_minute

    # LLM inference cost on every ticket.
    llm_cost = annual_volume * llm_cost_per_ticket

    engine_total_cost = auto_handling_cost + human_handling_cost + rework_cost + llm_cost
    net_savings = baseline_cost - engine_total_cost

    # Hours of agent capacity freed up.
    hours_saved = (baseline_cost - (auto_handling_cost + human_handling_cost)) / loaded_hourly_cost

    return {
        "annual_volume": int(annual_volume),
        "baseline_annual_cost": round(baseline_cost, 2),
        "engine_annual_cost": round(engine_total_cost, 2),
        "net_annual_savings": round(net_savings, 2),
        "savings_pct": round(net_savings / baseline_cost, 4) if baseline_cost else 0.0,
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
            "pretriage_time_saving": pretriage_time_saving,
        },
    }
