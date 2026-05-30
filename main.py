"""
main.py
───────
Experiment runner: compare LLM agents with and without the Algorithm 1 workflow.

Paper basis (Section 5):
  We replicate the core finding: LLM agents with the workflow scaffold reach
  more envy-free, Pareto-optimal deals than the baseline (no workflow).

Usage
-----
  python main.py              # run N games per condition (N from config.py)
  python main.py --dry-run    # use canned LLM responses (no API calls)
  python main.py --n 3        # override number of games per condition
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from game.environment import GameEnvironment
from agents.gemini_agent import GeminiAgent
from agents.heuristic_agent import HeuristicAgent
from llm.gemini_client import GeminiClient
from negotiation.protocol import GameResult, NegotiationProtocol
from game.metrics import MetricsReport
import config


# ─────────────────────────────────────────────
# Single game runner
# ─────────────────────────────────────────────

def run_one_game(
    seed: int,
    use_workflow: bool,
    client: GeminiClient,
    verbose: bool = False,
) -> GameResult:
    """Instantiate one random game and run it with two GeminiAgents."""
    env = GameEnvironment.setup_random(seed=seed, max_rounds=config.MAX_ROUNDS)

    agent_a = GeminiAgent(
        "A", env,
        use_workflow=use_workflow,
        client=client,
        belief_candidates=config.BELIEF_CANDIDATES,
        seed=seed,
    )
    agent_b = GeminiAgent(
        "B", env,
        use_workflow=use_workflow,
        client=client,
        belief_candidates=config.BELIEF_CANDIDATES,
        seed=seed + 1000,
    )

    protocol = NegotiationProtocol(agent_a, agent_b, env)
    result = protocol.run()

    if verbose:
        print(result.transcript)

    return result


# ─────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────

def aggregate(results: List[GameResult]) -> Dict:
    """Compute summary statistics across a list of game results."""
    n = len(results)
    deals = [r for r in results if r.deal_reached]
    n_deals = len(deals)

    if n_deals == 0:
        return {
            "n": n, "n_deals": 0, "deal_rate": 0.0,
            "avg_social_welfare": 0.0, "avg_rounds": sum(r.rounds_used for r in results) / n,
            "pct_pareto": 0.0, "pct_envy_free": 0.0, "pct_proportional": 0.0,
            "pct_above_threat": 0.0,
        }

    m: List[MetricsReport] = [r.metrics for r in deals]  # type: ignore[misc]
    return {
        "n": n,
        "n_deals": n_deals,
        "deal_rate": n_deals / n,
        "avg_social_welfare": sum(x.social_welfare for x in m) / n_deals,
        "avg_rounds": sum(r.rounds_used for r in results) / n,
        "pct_pareto": sum(1 for x in m if x.pareto_optimal) / n_deals,
        "pct_envy_free": sum(1 for x in m if x.envy_free) / n_deals,
        "pct_proportional": sum(1 for x in m if x.proportional) / n_deals,
        "pct_above_threat": sum(1 for x in m if x.above_threat) / n_deals,
    }


# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

def print_game_line(seed: int, result: GameResult, condition: str) -> None:
    """Print a one-line summary of a single game result."""
    if result.deal_reached and result.metrics:
        m = result.metrics
        print(
            f"  Game seed={seed:>3} | DEAL  in {result.rounds_used:>2} rounds | "
            f"SW={m.social_welfare:>5.1f} | "
            f"PO={'Y' if m.pareto_optimal else 'N'} "
            f"EF={'Y' if m.envy_free else 'N'} "
            f"PR={'Y' if m.proportional else 'N'}"
        )
    else:
        print(
            f"  Game seed={seed:>3} | NO DEAL ({result.termination:<7}) | "
            f"rounds={result.rounds_used:>2} | threat point activated"
        )


def print_comparison_table(
    baseline_stats: Dict,
    workflow_stats: Dict,
) -> None:
    """Print the final side-by-side comparison table."""
    W = 18

    def pct(v: float) -> str:
        return f"{v * 100:>5.0f}%"

    def fmt(v: float, is_pct: bool = False) -> str:
        return pct(v) if is_pct else f"{v:>6.2f}"

    sep = "=" * 82
    print(f"\n{sep}")
    print("  COMPARISON TABLE  (Lesson 12: LLM baseline vs. Algorithm 1 workflow)")
    print(sep)
    print(
        f"  {'Condition':<{W}} | {'Deals':>6} | {'Rate':>6} | "
        f"{'Avg SW':>7} | {'Avg Rnd':>7} | {'Pareto':>7} | {'EF':>7} | {'Prop':>7}"
    )
    print("-" * 82)
    for label, s in [("Without Workflow", baseline_stats), ("With Workflow   ", workflow_stats)]:
        deals_str = f"{s['n_deals']}/{s['n']}"
        print(
            f"  {label:<{W}} | {deals_str:>6} | {pct(s['deal_rate']):>6} | "
            f"{s['avg_social_welfare']:>7.2f} | {s['avg_rounds']:>7.2f} | "
            f"{pct(s['pct_pareto']):>7} | {pct(s['pct_envy_free']):>7} | "
            f"{pct(s['pct_proportional']):>7}"
        )
    print(sep)
    print()
    print("  Metric legend:")
    print("    SW   = Social Welfare (sum of both agents' utilities) — higher is better")
    print("    PO   = Pareto Optimal (no deal could make someone strictly better off)")
    print("    EF   = Envy-Free (neither agent prefers the other's bundle)")
    print("    Prop = Proportional (each agent receives >= 1/2 of their max value)")
    print(f"\n  Lesson 10 note: any deal above 0 beats the Threat Point d=(0,0).")
    print(sep)


# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────

def save_results(
    baseline_results: List[GameResult],
    workflow_results: List[GameResult],
    baseline_stats: Dict,
    workflow_stats: Dict,
    out_dir: Path,
) -> None:
    """Write full transcripts and aggregated stats to the results/ folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Aggregated stats JSON
    stats_path = out_dir / f"stats_{ts}.json"
    stats_path.write_text(
        json.dumps({"baseline": baseline_stats, "workflow": workflow_stats}, indent=2),
        encoding="utf-8",
    )

    # Full transcripts
    for label, results in [("baseline", baseline_results), ("workflow", workflow_results)]:
        for i, result in enumerate(results):
            t_path = out_dir / "transcripts" / f"{label}_game{i}_{ts}.txt"
            t_path.parent.mkdir(parents=True, exist_ok=True)
            t_path.write_text(result.transcript + "\n" + result.summary(), encoding="utf-8")

    print(f"\n  Results saved to: {out_dir}/")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Game Theory Negotiation Experiment")
    parser.add_argument("--n", type=int, default=config.RUNS_PER_CONDITION,
                        help="Number of games per condition (default from config.py)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting RNG seed (games use seed, seed+1, ...)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use canned LLM responses — no API calls")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full transcript for each game")
    args = parser.parse_args()

    n = args.n
    seed_start = args.seed
    seeds = list(range(seed_start, seed_start + n))

    sep = "=" * 62
    print(f"\n{sep}")
    print("  GAME-THEORETIC LLM NEGOTIATION — EXPERIMENT RUNNER")
    print("  Paper: 'Game-theoretic LLM: Agent Workflow for")
    print("          Negotiation Games' (Nov 2024)")
    print(f"  Games per condition : {n}")
    print(f"  Seeds               : {seeds[0]} – {seeds[-1]}")
    print(f"  Mode                : {'DRY RUN (no API)' if args.dry_run else 'LIVE (Gemini API)'}")
    print(sep)

    # One shared client to reuse the HTTP session and the log file
    client = GeminiClient(dry_run=args.dry_run)

    # ── Condition 1: WITHOUT workflow ────────────────────────────────
    print(f"\n[1/2] Running {n} games WITHOUT workflow (baseline)...\n")
    baseline_results: List[GameResult] = []
    for seed in seeds:
        print(f"  Starting game seed={seed}...", end=" ", flush=True)
        result = run_one_game(seed, use_workflow=False, client=client, verbose=args.verbose)
        print_game_line(seed, result, "baseline")
        baseline_results.append(result)

    # ── Condition 2: WITH workflow ───────────────────────────────────
    print(f"\n[2/2] Running {n} games WITH workflow (Algorithm 1)...\n")
    workflow_results: List[GameResult] = []
    for seed in seeds:
        print(f"  Starting game seed={seed}...", end=" ", flush=True)
        result = run_one_game(seed, use_workflow=True, client=client, verbose=args.verbose)
        print_game_line(seed, result, "workflow")
        workflow_results.append(result)

    # ── Aggregate & display ──────────────────────────────────────────
    baseline_stats = aggregate(baseline_results)
    workflow_stats = aggregate(workflow_results)
    print_comparison_table(baseline_stats, workflow_stats)

    # ── Save ─────────────────────────────────────────────────────────
    save_results(
        baseline_results, workflow_results,
        baseline_stats, workflow_stats,
        out_dir=Path("results"),
    )


if __name__ == "__main__":
    main()
