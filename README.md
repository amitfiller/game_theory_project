# 🎲 When the Student Surpasses the Teacher
### A Game-Theoretic Audit of LLM Negotiation

> A reproduction and **stress-test** of the paper *"Game-Theoretic LLM: Agent Workflow for Negotiation Games"* (2024) —
> final project for the course **Introduction to Game Theory & Decision Making** (Dr. Raz Lin).

Two LLM agents (powered by **Gemini 2.5-flash**) negotiate over a shared pool of resources (books, hats, balls).
The question: **Is a language model truly a rational player — and what does a structured "workflow" (Algorithm 1) add over a raw LLM?**

---

## 🏁 TL;DR — The Key Finding

We pitted two conditions head-to-head on the exact same model:

| Condition | Social Welfare | Pareto | Envy-Free | Avg. Rounds |
|------|:---:|:---:|:---:|:---:|
| **Baseline** (raw Gemini) | **24.1** | **90%** | **100%** | **2.4** |
| **Workflow** (full Algorithm 1) | 23.1 | 70% | 80% | 3.2 |

The **Baseline won on every metric**. Why: 2026-era models are already strong enough to reach near-optimal deals on their own (a **ceiling effect**), and the workflow's strict fairness enforcement sometimes even caused **deadlock** ending at (0, 0). This is a **well-reasoned negative result** — and that is precisely the contribution.

---

## ✅ Prerequisites

- **Python 3.10+** (the code uses `int | None` syntax; developed and tested on 3.12)
- **pip**
- *(live runs only)* a free **Gemini API** key — see [Step 3](#3-live-run-api-key).
- ⚠️ For a quick check you **do not need an API key at all** — see the `--dry-run` mode below.

All external dependencies are listed in [`requirements.txt`](requirements.txt):
`google-genai`, `pydantic`, `python-dotenv` (+ `pytest` for tests). Everything else is part of the Python standard library.

---

## 🚀 Installation & Running (Instructor Guide)

### 1. Clone and install dependencies

```bash
git clone https://github.com/amitfiller/game_theory_project.git
cd game_theory_project

# Recommended: a virtual environment
python -m venv .venv

# Activate it:
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Quick check without an API key (recommended starting point) ⭐

The `--dry-run` mode runs the **entire pipeline** with canned responses — no network calls, no key required. Perfect for verifying everything works:

```bash
python main.py --dry-run
```

You can also run the core tests (pure Python, zero API calls):

```bash
python test_core.py        # unit tests for the game engine and metrics
python test_game_loop.py   # integration test — a full game between two heuristic agents
# or, if pytest is installed:
pytest
```

### 3. Live run: API key

1. Get a free key from Google AI Studio: **https://aistudio.google.com/app/apikey**
2. Create a file named `.env` in the project root (it is already in `.gitignore` and will not be pushed to git):

   ```env
   GEMINI_API_KEY=your_key_here
   ```

3. Run the full experiment:

   ```bash
   python main.py
   ```

   This runs N games per condition (baseline / workflow), prints a **comparison table**, and saves everything to `results/`.

---

## ⚙️ Flags & Options

```bash
python main.py --dry-run     # no API calls (canned responses)
python main.py --n 3         # 3 games per condition (default: 5)
python main.py --seed 42     # starting seed (for reproducibility)
python main.py --verbose     # print the full transcript for each game
```

Parameters can also be controlled via the `.env` file (see [`config.py`](config.py)):
`GEMINI_MODEL`, `MAX_ROUNDS`, `RUNS_PER_CONDITION`, `BELIEF_CANDIDATES`, `LLM_TEMPERATURE`, and more.

---

## 📂 Project Structure

```
game_theory_project/
├── main.py                  # entry point — runs the full experiment
├── config.py                # central configuration (model, key, parameters)
│
├── game/                    # game engine (pure Python, no LLM)
│   ├── allocation.py        #   Item · Valuation · Allocation
│   ├── environment.py       #   GameEnvironment — single source of truth
│   └── metrics.py           #   fairness metrics: EF · Proportional · Pareto · Threat
│
├── agents/                  # the agents
│   ├── base_agent.py        #   abstract interface for every agent
│   ├── beliefs.py           #   BeliefState — Bayesian update about the opponent
│   ├── heuristic_agent.py   #   deterministic agent (baseline + fallback)
│   └── gemini_agent.py      #   the main Gemini-powered agent (+ Envy-Gate)
│
├── negotiation/             # the negotiation protocol
│   ├── messages.py          #   Proposal · Accept · Reject · CounterProposal
│   └── protocol.py          #   NegotiationProtocol — the Algorithm 1 loop
│
├── llm/
│   └── gemini_client.py     # Gemini API wrapper (structured output + retry)
│
├── prompts/
│   └── workflow_prompts.py  # PromptBuilder — baseline vs. workflow prompts
│
├── results/                 # run outputs (stats + transcripts)
│
├── test_core.py             # unit tests (no API)
├── test_game_loop.py        # integration test (no API)
└── requirements.txt
```

---

## 📊 Outputs (`results/`)

After a run, the following are produced:
- `results/stats_<timestamp>.json` — aggregated statistics (baseline vs. workflow).
- `results/transcripts/<condition>_game<i>_<timestamp>.txt` — full human-readable transcript per game.
- `results/transcripts/session_<timestamp>.jsonl` — raw log of every API call (prompt + response).

---

## 🧠 Core Design Principles

1. **Math judges, the LLM plays** — all Bayesian, utility, and metric computations live in pure Python; the LLM makes strategic decisions only, never acting as the referee.
2. **Modular architecture** — the game engine is fully decoupled from the agents; `GeminiAgent` ↔ `HeuristicAgent` are swappable behind a single interface.
3. **Operational robustness** — Structured Outputs (guaranteed valid JSON), Smart Retry (temperature bump to escape a loop), and a heuristic Fallback that ensures the run never crashes.
4. **Reproducibility** — fixed seeds, `temperature=0`, and all transcripts archived as ground truth.

---

## 📜 Academic Background

This project is based on the paper:
> Hua et al., *"Game-Theoretic LLM: Agent Workflow for Negotiation Games"*, 2024.

The base game is **"Deal or No Deal"** (Lewis et al., 2017). We reproduced the setup (items whose values sum to 10, incomplete information) and extended it into a **critical audit** with mathematically enforced fairness (the Envy-Gate) tested on a newer model generation.

> ℹ️ **Note on reproduction fidelity:** several components in the code are *our own* design choices, not the paper's — the 10-round cap, the (0, 0) threat point, and the strict Envy-Gate. In the original paper the algorithm runs with no round limit and converges purely through Bayesian belief updates.

---

## 👥 Presenters

**Amit Filler** · **Elad Dayan** — Game Theory course, supervised by **Dr. Raz Lin**.
</content>
