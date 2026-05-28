"""
negotiation/protocol.py
───────────────────────
NegotiationProtocol: orchestrates Algorithm 1 from the paper (Section 5).

Theory basis (Lesson 6 – Decision Trees / Lesson 10 – Negotiations):
  The negotiation is a finite extensive-form game:
    - Alternating moves: at each round one agent proposes, the other responds.
    - Three terminal actions: Accept (deal), Reject (no deal), or max-rounds exceeded.
    - Threat point d=(0,0) is activated on any non-deal termination.
    - Bayesian belief updates occur after every observed proposal.

Algorithm 1 (paper Section 5) loop:
  round = 1
  proposer, responder = agentA, agentB
  while round <= T:
      proposal  = proposer.propose(history)
      response  = responder.respond(proposal, history)
      if Accept  → return Deal
      if Reject  → return NoDeal
      if Counter → update proposer's belief; swap roles; round += 1
  return NoDeal (timeout)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from game.allocation import Allocation
from game.environment import GameEnvironment
from game.metrics import FairnessMetrics, MetricsReport
from agents.base_agent import AbstractAgent, History
from negotiation.messages import (
    Accept, CounterProposal, Proposal, Reject, Response
)


# ─────────────────────────────────────────────
# GameResult
# ─────────────────────────────────────────────

@dataclass
class GameResult:
    """
    Full record of a completed negotiation run.

    Attributes
    ----------
    deal_reached   : True iff agents agreed before round T+1
    final_proposal : the accepted Proposal (None if no deal)
    rounds_used    : how many rounds were played
    termination    : "deal" | "reject" | "timeout"
    metrics        : FairnessMetrics report (None if no deal)
    transcript     : human-readable log of every move
    history        : raw (round, proposer_id, proposal, response) records
    """
    deal_reached: bool
    final_proposal: Optional[Proposal]
    rounds_used: int
    termination: str                        # "deal" | "reject" | "timeout"
    metrics: Optional[MetricsReport]
    transcript: str
    history: History = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Game Result ===",
            f"  Outcome       : {'DEAL' if self.deal_reached else 'NO DEAL'} ({self.termination})",
            f"  Rounds used   : {self.rounds_used}",
        ]
        if self.deal_reached and self.metrics:
            lines.append(f"  {self.metrics.summary()}")
        else:
            lines.append("  Utilities     : threat point activated — both agents receive 0")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ─────────────────────────────────────────────
# NegotiationProtocol
# ─────────────────────────────────────────────

class NegotiationProtocol:
    """
    Implements the Algorithm 1 negotiation loop from the paper (Section 5).

    Responsibilities
    ----------------
    - Orchestrate turn-taking between two agents.
    - Enforce the max_rounds horizon.
    - Trigger Bayesian belief updates on each agent after they observe a proposal.
    - Log a full human-readable transcript for inspection and presentation.
    - Return a GameResult with metrics for any deal that is reached.

    This class is intentionally agent-agnostic: it works with any pair of
    AbstractAgent subclasses (HeuristicAgent, GeminiAgent, etc.).
    """

    def __init__(
        self,
        agent_a: AbstractAgent,
        agent_b: AbstractAgent,
        env: GameEnvironment,
    ) -> None:
        """
        Parameters
        ----------
        agent_a : the agent that proposes first (round 1 proposer)
        agent_b : the agent that responds first (round 1 responder)
        env     : the shared GameEnvironment
        """
        if agent_a.agent_id == agent_b.agent_id:
            raise ValueError("Both agents have the same agent_id — they must differ.")
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.env = env
        self._log_lines: List[str] = []

    # ── main loop ─────────────────────────────────────────────────────

    def run(self) -> GameResult:
        """
        Execute the full negotiation loop and return a GameResult.

        This is the top-level entry point.  Call once per game instance.
        """
        history: History = []
        proposer: AbstractAgent = self.agent_a
        responder: AbstractAgent = self.agent_b
        self._log_lines = []

        self._log("=== Negotiation Start ===")
        self._log(f"    Item pool : {self.env.item_pool}")
        self._log(f"    Max rounds: {self.env.max_rounds}")
        self._log("")

        for round_num in range(1, self.env.max_rounds + 1):

            self._log(f"--- Round {round_num} | Proposer: {proposer.agent_id} ---")

            # ── Step 1: proposer generates an offer ──────────────────
            proposal = proposer.propose(history)
            self._log(f"  PROPOSAL  : {proposal}")
            if proposal.reasoning:
                self._log(f"  reasoning : {textwrap.shorten(proposal.reasoning, 120)}")

            # ── Step 2: responder observes and reacts ────────────────
            response = responder.respond(proposal, history)
            self._log(f"  RESPONSE  : {response}")
            if hasattr(response, "reasoning") and response.reasoning:
                self._log(f"  reasoning : {textwrap.shorten(response.reasoning, 120)}")

            # Record in history
            history.append((round_num, proposer.agent_id, proposal, response))

            # ── Step 3: handle terminal responses ────────────────────
            if isinstance(response, Accept):
                return self._finish_deal(proposal, round_num, history)

            if isinstance(response, Reject):
                self._log(f"\n  !! {responder.agent_id} REJECTED — no deal (threat point: both get 0)")
                return self._finish_no_deal("reject", round_num, history)

            # ── Step 4: counter-proposal — update beliefs, swap roles ─
            if isinstance(response, CounterProposal):
                # The PROPOSER now observes what the responder wants for themselves
                # This is the Bayesian signal: what did the opponent allocate to themselves?
                counter_self_bundle = response.counter.bundle_of(responder.agent_id)
                proposer.belief.update(counter_self_bundle)
                self._log(
                    f"  [Belief update] {proposer.agent_id} observes "
                    f"{responder.agent_id}'s self-allocation: {counter_self_bundle}"
                )

                # Swap roles for next round
                proposer, responder = responder, proposer
                self._log("")

        # ── Timeout: max rounds reached without agreement ─────────────
        self._log(f"\n  !! Max rounds ({self.env.max_rounds}) reached — no deal (timeout)")
        return self._finish_no_deal("timeout", self.env.max_rounds, history)

    # ── result builders ───────────────────────────────────────────────

    def _finish_deal(
        self, proposal: Proposal, round_num: int, history: History
    ) -> GameResult:
        """Build a GameResult for a successful deal and compute fairness metrics."""
        alloc = Allocation(bundles=proposal.bundles)
        metrics = FairnessMetrics.evaluate(alloc, self.env)

        self._log(f"\n  !! DEAL reached in round {round_num}!")
        self._log(metrics.summary())

        return GameResult(
            deal_reached=True,
            final_proposal=proposal,
            rounds_used=round_num,
            termination="deal",
            metrics=metrics,
            transcript="\n".join(self._log_lines),
            history=history,
        )

    def _finish_no_deal(
        self, reason: str, round_num: int, history: History
    ) -> GameResult:
        """Build a GameResult for a failed negotiation (threat point activated)."""
        return GameResult(
            deal_reached=False,
            final_proposal=None,
            rounds_used=round_num,
            termination=reason,
            metrics=None,
            transcript="\n".join(self._log_lines),
            history=history,
        )

    # ── logging ───────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._log_lines.append(msg)
