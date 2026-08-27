# 🤖 AI-Powered Support Ticket Triage & Routing Engine
### An LLM triage engine that reads customer messages, sorts them, drafts replies, and knows when to hand a ticket to a human — with a trained ML baseline benchmarked on the **real, public BANKING77 dataset**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange?logo=jupyter)](https://jupyter.org/)
[![Claude](https://img.shields.io/badge/Claude-LLM%20Engine-8e44ad)](https://www.anthropic.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Baseline-f7931e?logo=scikitlearn)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Data-BANKING77%20(real)-27ae60)](https://github.com/PolyAI-LDN/task-specific-datasets)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io%2Fjordanshamu-2c3e50)](https://datascienceportfol.io/jordanshamu)

> **AI Automation Project** in Jordan Shamukiga's Data Science & Analytics Portfolio
> Analyst: [Jordan Shamukiga](https://github.com/jordanshamu) · [datascienceportfol.io/jordanshamu](https://datascienceportfol.io/jordanshamu)

> **The data here is real.** Every number below is computed on **BANKING77** — 13,083 real online-banking customer-service messages, each labelled with one of 77 intents, released by PolyAI ([Casanueva et al., ACL 2020](https://github.com/PolyAI-LDN/task-specific-datasets), CC-BY-4.0). Nothing is synthetic. The loader downloads it on first run.

---

## Business Context

Before anyone answers a support ticket, someone has to read it, work out what it's about, decide how urgent it is, and send it to the right place. Multiply that across an inbox and triage becomes the bottleneck — the step that decides how fast customers actually get helped.

This project automates that step and then does the honest thing: it tests the automation on messages I didn't write. For each inbound ticket the LLM engine returns, in a **single call**:

- **Intent** — one of the 77 real BANKING77 intents (e.g. `card_arrival`, `pending_transfer`, `compromised_card`)
- **Urgency** — low · medium · high · critical
- **Sentiment** — negative · neutral · positive
- **Structured entities** — amounts, card/account references, dates
- **A drafted first reply** the agent can edit and send

A confidence gate then decides what ships on its own and what goes to a person. High-stakes intents — anything touching fraud, loss, or money at risk — always go to a human, confidence notwithstanding. That gate is the whole reason this is deployable rather than a demo.

**One honesty note up front.** BANKING77 labels only the **intent**. It has no urgency, sentiment, or entity ground truth. So the *scored* benchmark is intent classification; the LLM's other four outputs are shown as capabilities, not graded against labels the dataset doesn't have. I'd rather report one real number than four invented ones.

---

## Key Results

### Trained ML baseline on the real test set (BANKING77, 3,080 held-out messages, 77 intents)

| Metric | TF-IDF + Logistic Regression |
|---|---|
| Intent accuracy | **0.848** |
| Macro-F1 (all 77 intents) | **0.848** |
| Weighted-F1 | **0.848** |
| Mean predicted confidence | 0.678 |

These are real, recomputed at runtime into `reports/triage_metrics.json`. A bag-of-words logistic regression getting **~85% right across 77 fine-grained classes** is a genuinely strong control — and it sets a high bar for whether an LLM is worth its per-call cost on the *routing* task specifically.

### The LLM vs ML head-to-head: wired, but not run in this build

> **I did not fabricate an LLM score.** The zero-shot Claude side of the benchmark needs a live API call per ticket, and this repo was built in an environment with **no `ANTHROPIC_API_KEY`**. The LIVE path is fully implemented — real 77-intent prompt, JSON contract, cost estimator (`src/llm_classifier.py`) — and the notebook runs the head-to-head automatically **the moment a key is present**. Until then, the LLM's intent accuracy is deliberately left blank rather than filled with the offline stub's number, because on 77-way real text that stub is a keyword heuristic, **not** a language model, and scoring it as "Claude" would manufacture a false result. Run LIVE to populate this row. (Cost to score the full 3,080-ticket test set is roughly a few dollars at Sonnet-class pricing — see `estimate_live_cost`.)

### The real finding: the errors aren't random — they're near-synonyms

The baseline's ~15% error rate is not spread evenly. It concentrates in tight clusters of intents that are near-synonymous even to a human reader:

- **Identity trio** — `why_verify_identity` → `verify_my_identity` (10 tickets), `unable_to_verify_identity` → `verify_my_identity` (6), and back the other way. These three intents are the model's single worst region.
- **Transfer states** — `pending_transfer` ↔ `failed_transfer` ↔ `transfer_timing` blur together.
- **Card ordering** — `order_physical_card` → `getting_spare_card` (6).
- **Virtual cards** — `virtual_card_not_working` → `get_disposable_virtual_card` (6).

The three lowest-F1 intents are `verify_my_identity` (0.68), `pending_transfer` (0.68), and `getting_spare_card` (0.70); the cleanest are `passcode_forgotten` (1.00), `apple_pay_or_google_pay` (0.99), and `card_about_to_expire` (0.99). My read: **this is as much a labelling-taxonomy problem as a modelling problem.** When three intents describe the same underlying question ("why must I verify," "how do I verify," "I can't verify"), a bag-of-words model has almost no signal to separate them — and honestly, neither would many human agents. That's also the most testable case *for* an LLM: semantic disambiguation of near-duplicate intents is exactly where a language model might pull ahead, which is the first thing I'd measure once the LIVE key is in.

### Automation & Business Impact — *illustrative model, not a measured result*

The dollar figures below come from a unit-economics model (`src/business_impact.py`) that takes the **real** automation rate and routing accuracy from the trained baseline and combines them with an **assumed** 12,000-ticket/month volume and standard handling times. Treat it as a framework to drop your own numbers into — not as savings this project banked.

| Metric | Value | Real or modeled? |
|---|---|---|
| Tickets auto-handled at a 0.75 gate | **42.8%** | **real** (baseline confidence on real test set) |
| Routing accuracy on the automated stream | **97.9%** | **real** |
| High-stakes tickets forced to a human | 583 of 3,080 | **real** |
| Modeled annual baseline cost (all-manual) | ~$691K | assumed volume |
| Modeled net annual savings with the engine | **~$263K (≈38%)** | modeled |
| Modeled agent-hours freed per year | **~8,530** | modeled |

**On the 0.75 gate:** I didn't pick it to maximise coverage. Confidence separates right from wrong cleanly on this data (viz 05) — above 0.75 the automated stream holds ~98% accuracy. Loosening the gate automates more but starts admitting the band where the model's mistakes live; I'd rather defend a smaller, cleaner automated stream to an ops lead than a bigger one I keep apologising for. The threshold sweep (viz 07) lets anyone move that line. The savings come almost entirely from **agent time recovered**, not from the model being cheap — so the case doesn't collapse if LLM pricing moves.

---

## The Architecture (Why It Works)

```
                 ┌─────────────────────────────────────────────┐
 Inbound ticket  │  LLM Triage Engine (Claude, structured JSON) │
 ──────────────► │  intent · urgency · sentiment · entities ·   │
                 │  confidence · drafted reply                  │
                 └───────────────┬─────────────────────────────┘
                                 │   (intent + confidence drive routing;
                                 │    the trained baseline can drive it too)
                    ┌────────────▼────────────┐
                    │   Confidence Gate        │
                    │  not high-stakes  AND    │
                    │  confidence ≥ 0.75       │
                    └──────┬───────────┬───────┘
                      auto │           │ escalate
                           ▼           ▼
              Route + queue draft   Human agent
              for one-click send    (full handling)
```

High-stakes intents (fraud, loss, security, money at risk — 15 of the 77, listed in `src/data_loader.py`) are **never** auto-resolved regardless of confidence. The cost of being wrong on `compromised_card` or `transfer_not_received_by_recipient` is measured in real harm and churn, not seconds of agent time.

---

## Methodology

### 1. Real Data (BANKING77)
`src/data_loader.py` downloads the dataset from PolyAI's GitHub, caches it under `data/raw/`, and applies the light cleaning it actually needs. The data is unusually clean — no nulls, no exact duplicates, no train/test text overlap, and every message maps to exactly one intent — so cleaning is limited to stripping whitespace (9 rows) and dropping anything empty. Train is mildly imbalanced (35–187 messages/intent); the test set is balanced at 40/intent. Messages are short (median ~10 words). **I did not invent fields the data lacks.**

### 2. Traditional ML Baseline
TF-IDF (uni + bigram) → class-balanced Logistic Regression. The pre-LLM standard and the control in the benchmark. It predicts intent and nothing else.

### 3. LLM Triage Engine
An engineered system prompt lists all 77 intents and forces a strict JSON contract (intent, urgency, sentiment, confidence, entities, draft reply). Runs against the real **Claude Messages API** in LIVE mode. Offline, it returns a schema-valid stub with `intent=None` so the pipeline's *shape* runs anywhere — but that stub is never scored as the LLM, because it isn't one.

### 4. Evaluation
Identical scoring for both systems on the same held-out split — accuracy, macro/weighted F1, per-intent precision/recall, confusion analysis. Scored on **intent only**, the one dimension with ground truth.

### 5. Confidence Gating
A threshold sweep (0.50 → 0.90) traces the coverage-vs-accuracy curve on the real test set and sets the operating point. High-stakes intents skip automation entirely.

### 6. Business ROI Model
A transparent unit-economics model turns the **real** automation rate and routing accuracy into annual cost, agent hours, and LLM spend. Every assumption lives in one file — the figures are illustrative on an assumed volume and meant to be re-run with a real operation's own inputs.

---

## Visualisations

All plots auto-save to `visualizations/` on notebook run. Every figure below is drawn from the real data.

| # | File | Description |
|---|---|---|
| 01 | `01_intent_distribution.png` | Top-20 of 77 intents by training volume |
| 02 | `02_text_length_distribution.png` | Message length (real tickets) |
| 03 | `03_baseline_confusion_top_pairs.png` | The 15 intent pairs the baseline most often confuses |
| 04 | `04_baseline_per_intent_f1.png` | Per-intent F1: 10 hardest vs 10 easiest |
| 05 | `05_baseline_confidence_distribution.png` | Confidence separates correct from incorrect (justifies the gate) |
| 06 | `06_capability_comparison.png` | Output coverage: one LLM call vs trained classifier (conceptual) |
| 07 | `07_confidence_threshold_tradeoff.png` | Coverage vs accuracy across gating thresholds (real) |
| 08 | `08_automation_funnel.png` | Inbound → not high-stakes → confident → correctly routed |
| 09 | `09_roi_waterfall.png` | Modeled annual economics: baseline vs engine vs net savings |
| 10 | `10_intent_family_confusion.png` | Near-synonym clusters: identity & transfer intents |

---

## Project Structure

```
ai-support-triage-engine/
├── data/
│   ├── raw/                        ← BANKING77 CSVs (downloaded on first run; git-ignored)
│   └── processed/                  ← test_predictions.csv (real predictions, regenerated on run)
├── notebooks/
│   └── ai_ticket_triage.ipynb      ← Main analysis notebook (fully executed)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              ← Downloads/caches/cleans real BANKING77 (+ high-stakes set)
│   ├── llm_classifier.py           ← Claude API engine + 77-intent prompt + honest offline stub
│   ├── ml_baseline.py              ← TF-IDF + Logistic Regression baseline
│   ├── evaluation.py               ← Shared scoring & confusion analysis
│   └── business_impact.py          ← Confidence gating + ROI model
├── reports/
│   ├── executive_summary.md        ← Non-technical stakeholder summary
│   └── triage_metrics.json         ← All real metrics (regenerated on run)
├── visualizations/                 ← 10 figures (auto-saved on run)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Lab or Jupyter Notebook
- Internet access on first run (the loader downloads BANKING77 from GitHub)

### Installation

```bash
git clone https://github.com/jordanshamu/ai-support-triage-engine.git
cd ai-support-triage-engine

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter lab notebooks/ai_ticket_triage.ipynb
```

Run all cells top-to-bottom. On first run the loader downloads BANKING77 to `data/raw/`; figures save to `visualizations/`, predictions to `data/processed/`, and metrics to `reports/triage_metrics.json`.

### Two Execution Modes

**OFFLINE (default — zero key):** The full pipeline — real data, trained baseline, real metrics, gating, ROI, all 10 figures — runs with no API key. Only the *LLM* side of the head-to-head is skipped, and it's skipped honestly (no invented number).

**LIVE (real Claude API):** Set your key and the notebook runs the zero-shot LLM head-to-head against the same held-out tickets:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS/Linux
pip install anthropic
jupyter lab notebooks/ai_ticket_triage.ipynb
```

The notebook auto-detects the mode. The trained-baseline numbers in this repo are **real regardless of mode**; the LLM comparison populates only in LIVE mode.

---

## Technical Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML baseline, vectorisation, evaluation |
| anthropic | Claude API client (LIVE mode) |
| matplotlib | Visualisation |
| jupyter | Interactive notebook environment |

---

## Skills Demonstrated

- **Working with real, public benchmark data** — sourcing, honest EDA, cleaning decisions, licensing
- **NLP & text classification** — TF-IDF, supervised modelling, 77-class evaluation
- **ML vs LLM benchmarking** — a fair head-to-head design with a clear decision rule and no fabricated numbers
- **Production thinking** — human-in-the-loop gating, confidence thresholds, high-stakes guardrails
- **Business analytics** — transparent ROI modelling, real-vs-assumed labelling, executive communication
- **Analytical honesty** — reporting only what the data supports, and naming the gaps plainly

---

## Related Projects

- **Customer Churn Prediction** — end-to-end ML pipeline, cost-benefit threshold optimisation, SHAP explainability
- **Customer Segmentation & Cohort Analysis** — RFM, K-Means, CLV estimation
- **Marketing A/B Test Analysis** — [GitHub](https://github.com/jordanshamu/Marketing-A-B-Test-Analysis) — 588K users, 42.5% conversion lift

---

## Author

**Jordan Shamukiga — Data Analyst · Business Analyst · Data Scientist**

[![GitHub](https://img.shields.io/badge/GitHub-jordanshamu-181717?logo=github)](https://github.com/jordanshamu)
[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io-2c3e50)](https://datascienceportfol.io/jordanshamu)

---

## Data Source & License

BANKING77 is released by PolyAI under CC-BY-4.0: Casanueva, Temčinas, Gerz, Henderson, Vulić, *"Efficient Intent Detection with Dual Sentence Encoders"* (NLP4ConvAI @ ACL 2020). Source: https://github.com/PolyAI-LDN/task-specific-datasets

This project's code is licensed under the MIT License — see [LICENSE](LICENSE) for details.
