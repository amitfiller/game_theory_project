"""
metrics.py
──────────
Game-theoretic fairness and efficiency metrics for a two-agent allocation.

All four concepts below are taught directly in Lesson 10 (Negotiations &
Fair Division) and appear in the paper's evaluation criteria (Section 5).

Concepts implemented
────────────────────
1. Envy-Freeness (EF)
     Agent i does not envy agent j iff
       U_i(bundle_i) >= U_i(bundle_j)
     i.e., i prefers their own bundle at least as much as the other's,
     when evaluated with i's OWN valuation.

2. Proportionality (PROP)  ← Lesson 10, Fair Share Guarantee
     Agent i receives at least their "fair share":
       U_i(bundle_i) >= (1/n) × U_i(all_items)
     For n=2 agents this means each agent gets >= half of their total value.

3. Pareto Optimality (PO)
     An allocation is Pareto-optimal iff there is no other feasible allocation
     that makes at least one agent strictly better off without making the other
     strictly worse off.  We check this by exhaustive enumeration over all
     feasible allocations (tractable for small item sets).

4. Threat Point
     Lesson 10 / Nash Bargaining: the threat point (disagreement point) d
     is the payoff each player receives if no deal is reached.
     In the "Deal or No Deal" game: d = (0, 0).
     The threat point sets the baseline — any rational agreement must give
     each agent strictly more than their threat-point payoff.

Usage
─────
>>> from game.metrics import FairnessMetrics
>>> report = FairnessMetrics.evaluate(allocation, env)
>>> print(report)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from game.allocation import Allocation, Valuation
from game.environment import GameEnvironment


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class MetricsReport:
    """
    Container for all fairness/efficiency results for one allocation.

    Attributes
    ----------
    envy_free       : True iff neither agent envies the other's bundle
    proportional    : True iff both agents receive >= their fair share (1/n of total value)
    pareto_optimal  : True iff no other allocation Pareto-dominates this one
    above_threat    : True iff both agents receive strictly more than the threat point (0)
    utilities       : dict  agent_id → numeric utility under their private valuation
    fair_shares     : dict  agent_id → minimum utility required for proportionality
    threat_point    : the disagreement payoff vector (always (0, 0) for this game)
    social_welfare  : sum of all agents' utilities (utilitarian social welfare)
    """

    envy_free: bool
    proportional: bool
    pareto_optimal: bool
    above_threat: bool
    utilities: Dict[str, float]
    fair_shares: Dict[str, float]
    threat_point: Dict[str, float]
    social_welfare: float

    def summary(self) -> str:
        lines = [
            "--- Metrics Report ---",
            f"  Utilities       : {self.utilities}",
            f"  Fair shares     : {self.fair_shares}",
            f"  Threat point    : {self.threat_point}",
            f"  Social welfare  : {self.social_welfare:.2f}",
            f"  Envy-free       : {self.envy_free}",
            f"  Proportional    : {self.proportional}",
            f"  Pareto-optimal  : {self.pareto_optimal}",
            f"  Above threat pt : {self.above_threat}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────
# FairnessMetrics
# ─────────────────────────────────────────────

class FairnessMetrics:
    """
    Static methods for evaluating an Allocation against game-theoretic criteria.

    All methods are pure functions (no side effects, no state).
    They can therefore be unit-tested independently of any LLM or agent logic.
    """

    # ── main entry point ──────────────────────────────────────────────

    @staticmethod
    def evaluate(allocation: Allocation, env: GameEnvironment) -> MetricsReport:
        """
        Run all metrics and return a MetricsReport.

        Parameters
        ----------
        allocation : the allocation to evaluate
        env        : the GameEnvironment (needed for valuations + item pool)
        """
        agent_ids = env.agent_ids
        valuations = {a: env.get_valuation(a) for a in agent_ids}
        item_pool = env.item_pool

        # ── 1. Utilities ──
        utilities = {
            a: allocation.utility_for(a, valuations[a])
            for a in agent_ids
        }

        # ── 2. Threat Point  (Lesson 10 – Nash Bargaining) ──
        # d_i = 0 for all i: no deal → zero payoff.
        threat_point = {a: 0.0 for a in agent_ids}

        # ── 3. Above Threat Point ──
        above_threat = all(
            utilities[a] > threat_point[a] for a in agent_ids
        )

        # ── 4. Fair Shares  (Lesson 10 – Proportionality) ──
        n = len(agent_ids)
        fair_shares = {
            a: valuations[a].max_possible_utility(item_pool) / n
            for a in agent_ids
        }

        # ── 5. Envy-Freeness ──
        envy_free = FairnessMetrics.check_envy_free(
            allocation, agent_ids, valuations
        )

        # ── 6. Proportionality ──
        proportional = FairnessMetrics.check_proportional(
            allocation, agent_ids, valuations, item_pool
        )

        # ── 7. Pareto Optimality ──
        pareto_optimal = FairnessMetrics.check_pareto_optimal(
            allocation, agent_ids, valuations, item_pool
        )

        # ── 8. Social Welfare ──
        social_welfare = sum(utilities.values())

        return MetricsReport(
            envy_free=envy_free,
            proportional=proportional,
            pareto_optimal=pareto_optimal,
            above_threat=above_threat,
            utilities=utilities,
            fair_shares=fair_shares,
            threat_point=threat_point,
            social_welfare=social_welfare,
        )

    # ── individual checks ─────────────────────────────────────────────

    @staticmethod
    def check_envy_free(
        allocation: Allocation,
        agent_ids: List[str],
        valuations: Dict[str, Valuation],
    ) -> bool:
        """
        Envy-Freeness (Lesson 10):
          Agent i does NOT envy agent j iff
            U_i(bundle_i) >= U_i(bundle_j)   for all j ≠ i

        Note: we evaluate BOTH bundles with agent i's OWN valuation — this
        captures "would i prefer to swap?" not "does i think j is happy?"
        """
        for i in agent_ids:
            for j in agent_ids:
                if i == j:
                    continue
                ui_own = valuations[i].utility_of(allocation.bundle_of(i))
                ui_other = valuations[i].utility_of(allocation.bundle_of(j))
                if ui_own < ui_other:
                    return False  # agent i envies agent j
        return True

    @staticmethod
    def check_proportional(
        allocation: Allocation,
        agent_ids: List[str],
        valuations: Dict[str, Valuation],
        item_pool: Dict[str, int],
    ) -> bool:
        """
        Proportionality / Fair Share Guarantee (Lesson 10):
          Agent i is proportionally satisfied iff
            U_i(bundle_i) >= (1/n) × U_i(all_items)

        For n=2: each agent must get at least half of their maximum
        possible utility if they owned everything.
        """
        n = len(agent_ids)
        for agent in agent_ids:
            val = valuations[agent]
            received = val.utility_of(allocation.bundle_of(agent))
            fair_share = val.max_possible_utility(item_pool) / n
            if received < fair_share:
                return False
        return True

    @staticmethod
    def check_pareto_optimal(
        allocation: Allocation,
        agent_ids: List[str],
        valuations: Dict[str, Valuation],
        item_pool: Dict[str, int],
    ) -> bool:
        """
        Pareto Optimality:
          An allocation X is Pareto-optimal iff there is no feasible allocation Y such that:
            U_i(Y) >= U_i(X) for all i, with strict inequality for at least one i.

        We enumerate all feasible allocations and check whether any one of them
        Pareto-dominates the given allocation.

        Time complexity: O(Π (count_k + 1)) — small for the 6-item Lewis pool.
        """
        current_utils = {
            a: allocation.utility_for(a, valuations[a]) for a in agent_ids
        }

        for alt in FairnessMetrics._enumerate_allocations(agent_ids, item_pool):
            alt_utils = {
                a: valuations[a].utility_of(alt.get(a, {})) for a in agent_ids
            }
            # Pareto-dominates: weakly better for all AND strictly better for one
            weakly_better = all(alt_utils[a] >= current_utils[a] for a in agent_ids)
            strictly_better = any(alt_utils[a] > current_utils[a] for a in agent_ids)
            if weakly_better and strictly_better:
                return False  # dominated → not Pareto-optimal
        return True

    # ── threat point helpers ──────────────────────────────────────────

    @staticmethod
    def threat_point_payoffs(agent_ids: List[str]) -> Dict[str, float]:
        """
        Return the disagreement / threat-point payoff vector.

        Lesson 10 – Nash Bargaining Solution:
          d = (d_A, d_B) is the outcome if players fail to reach agreement.
          In the Deal-or-No-Deal game: d_A = d_B = 0.
          Any rational deal must give each agent strictly more than d_i.
        """
        return {a: 0.0 for a in agent_ids}

    @staticmethod
    def exceeds_threat_point(
        utilities: Dict[str, float],
        agent_ids: List[str],
    ) -> bool:
        """Return True iff every agent's utility strictly exceeds their threat-point value."""
        threat = FairnessMetrics.threat_point_payoffs(agent_ids)
        return all(utilities[a] > threat[a] for a in agent_ids)

    # ── allocation enumerator (used by Pareto check) ──────────────────

    @staticmethod
    def _enumerate_allocations(
        agent_ids: List[str],
        item_pool: Dict[str, int],
    ) -> Iterator[Dict[str, Dict[str, int]]]:
        """
        Yield every feasible allocation as a dict  {agent_id: {item: count}}.

        For two agents, each item with count k can be split 0-k to agent A
        and (k - split) to agent B.  We iterate over all combinations.

        Example: book×3 → agent A can get 0, 1, 2, or 3 books.
        """
        items = list(item_pool.keys())
        ranges = [range(item_pool[item] + 1) for item in items]

        a, b = agent_ids[0], agent_ids[1]
        for combo in itertools.product(*ranges):
            alloc: Dict[str, Dict[str, int]] = {
                a: {items[i]: combo[i] for i in range(len(items))},
                b: {items[i]: item_pool[items[i]] - combo[i] for i in range(len(items))},
            }
            yield alloc
