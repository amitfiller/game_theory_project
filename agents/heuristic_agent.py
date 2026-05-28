"""
agents/heuristic_agent.py
─────────────────────────
A fully deterministic, LLM-free agent that uses BeliefState + Expected Utility
Maximisation to negotiate.

Purpose
-------
This agent serves two roles:
  1. Offline testing: proves the entire game pipeline runs correctly without
     any API calls (grading criterion #4 – Operational).
  2. Baseline comparison: represents a "rational but non-AI" player, useful
     for illustrating what the LLM workflow adds.

Strategy (Lesson 3 – Strategies & Expected Utility):
  Proposer strategy:
    Enumerate all feasible allocations.
    Filter to those that are "fair" to the opponent (expected utility for the
    opponent >= opponent's expected proportional fair share under our belief).
    Among the fair candidates, pick the one maximising OUR utility.
    If no fair allocation exists, relax to "above threat point" (opponent gets > 0),
    and if still nothing, just maximise own utility.

  Responder strategy:
    Accept iff the offered bundle gives us >= our own proportional fair share.
    Otherwise, generate a counter-proposal using the same proposer strategy.

Theory grounding:
  - Expected Utility from Lesson 3: EU_i(a) = Σ P(type) × U_i(a | type)
  - Fair Share from Lesson 10: >= (1/n) × max_utility_i
  - Threat Point from Lesson 10: d_i = 0 (no deal → zero payoff)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from game.allocation import Allocation
from game.environment import GameEnvironment
from game.metrics import FairnessMetrics
from agents.base_agent import AbstractAgent, History
from negotiation.messages import (
    Accept, CounterProposal, Proposal, Reject, Response
)


class HeuristicAgent(AbstractAgent):
    """
    Greedy-rational agent: maximises own expected utility subject to
    an envy-free / proportionality constraint on the opponent.

    Parameters
    ----------
    agent_id           : "A" or "B"
    env                : the shared GameEnvironment
    belief_candidates  : size of the Bayesian belief support (default 30)
    seed               : RNG seed for the BeliefState sampler
    accept_threshold   : fraction of own fair share required to accept
                         (1.0 = full fair share, 0.8 = 80% of fair share).
                         Lowering this makes the agent more willing to accept.
    """

    def __init__(
        self,
        agent_id: str,
        env: GameEnvironment,
        belief_candidates: int = 30,
        seed: Optional[int] = None,
        accept_threshold: float = 1.0,
    ) -> None:
        valuation = env.get_valuation(agent_id)
        super().__init__(
            agent_id=agent_id,
            valuation=valuation,
            env=env,
            belief_candidates=belief_candidates,
            seed=seed,
        )
        self.accept_threshold = accept_threshold

    # ── propose ───────────────────────────────────────────────────────

    def propose(self, history: History) -> Proposal:
        """
        Generate the best allocation for self that is still "fair" to the opponent.

        Algorithm
        ---------
        1. Enumerate every feasible allocation of the item pool.
        2. Score each: (my_utility, passes_fairness_filter).
        3. Select the highest-utility allocation that passes the fairness filter.
           Fall back to "above threat point" if no fair allocation exists,
           and to pure max-utility if even that fails (degenerate game).
        4. Package as a Proposal.
        """
        round_num = len(history) + 1
        opponent_id = self.env.opponent_of(self.agent_id)
        item_pool = self.item_pool

        # Expected fair share for the opponent under current belief
        opp_expected_fair_share = self._opponent_expected_fair_share()

        best_bundle: Optional[Dict[str, Dict[str, int]]] = None
        best_utility = -1.0
        fallback_bundle: Optional[Dict[str, Dict[str, int]]] = None
        fallback_utility = -1.0

        for alloc in FairnessMetrics._enumerate_allocations(
            self.env.agent_ids, item_pool
        ):
            my_bundle = alloc[self.agent_id]
            opp_bundle = alloc[opponent_id]

            my_utility = self.valuation.utility_of(my_bundle)
            opp_expected_utility = self.belief.calculate_expected_utility(opp_bundle)

            # Fairness filter: does opponent receive at least their expected fair share?
            opponent_is_satisfied = opp_expected_utility >= opp_expected_fair_share

            if opponent_is_satisfied and my_utility > best_utility:
                best_utility = my_utility
                best_bundle = alloc

            # Fallback: opponent at least above threat point (utility > 0)
            if opp_expected_utility > 0 and my_utility > fallback_utility:
                fallback_utility = my_utility
                fallback_bundle = alloc

        chosen = best_bundle or fallback_bundle
        if chosen is None:
            # Degenerate: just keep everything (should never happen with >1 item)
            chosen = {
                self.agent_id: dict(item_pool),
                opponent_id: {k: 0 for k in item_pool},
            }

        return Proposal(
            proposer_id=self.agent_id,
            bundles=chosen,
            round_num=round_num,
            reasoning=(
                f"Maximised own EU={self.valuation.utility_of(chosen[self.agent_id]):.1f} "
                f"with opponent expected EU="
                f"{self.belief.calculate_expected_utility(chosen[opponent_id]):.1f} "
                f"(fair share threshold={opp_expected_fair_share:.1f})"
            ),
        )

    # ── respond ───────────────────────────────────────────────────────

    def respond(self, proposal: Proposal, history: History) -> Response:
        """
        Accept if the offered bundle meets our acceptance threshold; counter otherwise.

        Lesson 10 – Responder Strategy:
          Accept iff  U_i(offered_bundle) >= accept_threshold × fair_share_i
          The threat point acts as a hard floor: never accept a utility of 0
          when a counter-offer might do better.

        Parameters
        ----------
        proposal : the incoming offer
        history  : prior game events (used by LLM agents; heuristic agent ignores it)
        """
        # Update own belief about the proposer based on what they kept
        proposer_bundle = proposal.bundle_of(proposal.proposer_id)
        self.belief.update(proposer_bundle)

        my_bundle = proposal.bundle_of(self.agent_id)
        my_utility = self.valuation.utility_of(my_bundle)
        threshold = self.accept_threshold * self.own_fair_share

        if my_utility >= threshold:
            return Accept(
                responder_id=self.agent_id,
                accepted_proposal=proposal,
                reasoning=(
                    f"Accepted: utility={my_utility:.1f} >= "
                    f"threshold={threshold:.1f} (fair_share={self.own_fair_share:.1f})"
                ),
            )

        # Counter-propose: generate the best offer from our perspective
        counter = self.propose(history)
        return CounterProposal(
            responder_id=self.agent_id,
            rejected_proposal=proposal,
            counter=counter,
            reasoning=(
                f"Rejected: utility={my_utility:.1f} < threshold={threshold:.1f}. "
                f"Counter-proposing."
            ),
        )

    # ── internal helpers ──────────────────────────────────────────────

    def _opponent_expected_fair_share(self) -> float:
        """
        Estimate the opponent's proportional fair share using our current belief.

        Lesson 3 + Lesson 10:
          expected_fair_share = E[U_opponent(all_items)] / n_agents
                              = belief.calculate_expected_utility(item_pool) / 2

        We use the belief's expected utility if the opponent received EVERYTHING
        as a proxy for their max possible utility, then halve it for n=2 agents.
        """
        n = len(self.env.agent_ids)
        eu_if_gets_everything = self.belief.calculate_expected_utility(self.item_pool)
        return eu_if_gets_everything / n
