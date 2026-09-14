# Executive Summary: AI-Powered Support Ticket Triage Engine
## Automating the Front Door of Customer Support — benchmarked on real data

**Prepared by:** Jordan Shamukiga — Data Analyst · Business Analyst · Data Scientist
**Date:** August 2026
**Audience:** Operations Leadership, Customer Support, Product

---

## The Business Problem

Every customer message — a card that hasn't arrived, a transfer stuck in "pending," a payment the customer doesn't recognise — has to be read, understood, prioritised, and routed before anyone can act on it. Today that triage happens by hand. It's slow, repetitive, and it sets the ceiling on how fast customers get help.

At roughly **12,000 tickets a month**, even a few minutes of triage per ticket runs into thousands of agent hours a year — most of it spent on routine messages that follow the same handful of patterns.

The question this project answers: can triage be automated reliably enough to trust, and what is that worth — tested on **real customer messages**, not ones I made up?

---

## What We Built and What We Tested It On

An AI engine that reads each inbound ticket and returns five things at once: what it's about (the intent), how urgent it is, how the customer feels, the key facts pulled out automatically, and a ready-to-send draft reply.

The part that makes it safe to deploy: the engine knows when *not* to act. A confidence gate auto-handles only routine, high-confidence tickets and sends everything it reads as ambiguous or high-stakes — anything touching fraud, loss, or money at risk — to a person. How often that safety net misses is measured, not assumed, and reported below.

The benchmark runs on **BANKING77**: 13,083 real online-banking customer-service messages, each labelled with one of 77 intents, published by PolyAI (ACL 2020). This matters — the results below are earned on messages written by real customers, not on synthetic data.

---

## The Key Findings

**1. A simple trained model is already strong — 84.8% accuracy across 77 fine-grained intents.** That's the traditional, pre-AI approach (a model trained on labelled history), and on the specific job of routing text into known categories it sets a high bar. On real data, the "old" tool is very good and very cheap.

**2. Its mistakes aren't random — they're near-synonyms.** The model's errors concentrate in tight clusters of intents that mean almost the same thing: three different "verify my identity" intents that bleed into each other, transfer states ("pending" vs "failed" vs "timing") that blur, two card-ordering intents that overlap. My honest read is that this is **as much a labelling-taxonomy problem as a model problem** — when three categories describe the same underlying question, no simple model can cleanly separate them, and neither could many agents. This is also the clearest place an LLM might help, because semantic nuance is its strength.

**3. The LLM-vs-model head-to-head is wired but not yet run.** Scoring the LLM side requires a live API call per ticket, and this build had no API key available. Rather than fill that number in with a stand-in and pretend, **I left it blank and built the pipeline so it runs the moment a key is added.** That's the honest state of it: the trained-model numbers are real; the LLM comparison is one run away.

So the recommendation isn't "buy the AI." It's: the trained model already handles routing well and cheaply; add the LLM for the things the model structurally can't do (urgency, sentiment, drafts) and to attack the near-synonym confusions — then measure whether it actually wins there before committing.

---

## The Financial Case — an Illustrative Model

**Read this as a framework, not a promise.** The automation rate and routing accuracy below are **real** (measured on the held-out test set). The dollar figures apply those real rates to an **assumed** 12,000-ticket/month volume and standard handling times documented in `src/business_impact.py` — a template to plug your own numbers into, not savings this project has banked.

| Measure | Result | Basis |
|---|---|---|
| Share of tickets safely auto-handled (0.75 gate) | ~42.8% | **real** |
| Routing accuracy on the automated stream | ~97.9% | **real** |
| Tickets *flagged* high-stakes and sent to a human | 583 of 3,080 | **real** |
| High-stakes tickets the flag missed | 2 of 600 (0.33%) | **real** |
| Modeled all-manual annual cost | ~$691,000 | assumed volume |
| Modeled net annual savings with the engine | **~$263,000 (about 38%)** | modeled |
| Modeled agent hours freed each year | **~8,530 hours** | modeled |

Nearly all of that comes from **agent time recovered**, not from the AI being cheap to run — so the case doesn't collapse if AI pricing shifts.

**One assumption to be aware of before quoting the savings.** Tickets the engine escalates still reach an agent already sorted and with a draft reply attached, so the model assumes those take **15% less time** to handle. Nobody has timed that yet, and it accounts for about **$59,000** of the $263,000. Strip it out entirely and the saving is **~$204,000 (about 29%)**, which is the number I'd plan against until it's measured.

---

## What This Project Does Not Cover

- **The LLM side hasn't been scored yet.** The head-to-head needs a live API key; this build ran without one. No LLM accuracy figure is claimed. Running it is the first thing on the list, not the last.
- **The high-stakes safety net is very good, not perfect.** It acts on what the model *thinks* a ticket is about, so it catches everything the model recognises as risky — and nothing it confidently misreads as routine. On the real test set that was 2 tickets out of 600, about 1 in 300. Both were mildly worded ("My card was not accepted"), which is precisely why they were missed. A keyword filter was tested as a backstop and dropped: it caught neither one. The fix is a second, risk-only check on the model's own uncertainty, and it should be built before this runs on fraud-adjacent queues.
- **The dollar figures rest on assumed volume.** The automation *rate* is real; the pricing around it is a model. Before anyone acts on the savings, run the engine on a sample of the organisation's own historical tickets and check it against how they were actually routed.
- **No live helpdesk integration.** Wiring into Zendesk, Salesforce, or Intercom is real engineering beyond this prototype.
- **No proof yet that AI drafts resolve tickets faster.** That's a causal claim and needs a controlled A/B test to earn.

---

## Recommended Next Steps

1. **Run the LLM head-to-head.** Add an API key and score zero-shot Claude against the trained model on the same test set — starting with the near-synonym clusters where the model is weakest. This is the open question the project is built to answer.
2. **Validate on our own tickets.** Run both models over a sample of historical messages and check routing against what actually happened.
3. **A/B test the drafted replies.** Measure resolution time and CSAT for AI-drafted vs. human-written first responses — prove the drafts help, don't assume it.
4. **Pilot where a mistake is cheap.** Start on low-risk intents, never on fraud/loss/identity. Learn on the safe queue first.

---

*Full methodology, model evaluation, and reproducible code in the project notebook.*
*Contact: Jordan Shamukiga | [datascienceportfol.io/jordanshamu](https://datascienceportfol.io/jordanshamu)*
