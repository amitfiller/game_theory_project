"""
agents/base_agent.py
────────────────────
Abstract interface that every agent (heuristic or LLM-based) must implement.

Design principle (Lesson 3 – Strategies):
  A strategy in an extensive-form game maps every information set (everything
  the agent has observed so far) to an action.  AbstractAgent enforces this
  contract:  given the history of play, produce a Proposal or a Response.

  This abstraction lets the NegotiationProtocol stay completely agnostic about
  whether it is talking to a heuristic engine or a Gemini API call — satisfying
  the "Operational" grading criterion (HeuristicAgent lets us demo without internet).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from game.allocation import Valuation
from game.environment import GameEnvironment
from agents.beliefs import BeliefState
from negotiation.messages import Proposal, Response


# History type: list of (round_num, proposer_id, Proposal, Response) tuples
History = List[Tuple[int, str, Proposal, Response]]


class AbstractAgent(ABC):
    """
    Base class for all negotiation agents.

    Subclasses must implement `propose` and `respond`.
    All shared state (agent_id, valuation, belief) lives here.

    Attributes
    ----------
    agent_id    : unique identifier, e.g. "A" or "B"
    valuation   : the agent's PRIVATE item valuations (known only to itself)
    belief      : Bayesian belief over the opponent's valuation (updated each round)
    env         : reference to the GameEnvironment (for item pool, etc.)
    """

    def __init__(
        self,
        agent_id: str,
        valuation: Valuation,
        env: GameEnvironment,
        belief_candidates: int = 30,
        seed: int | None = None,
    ) -> None:
        self.agent_id: str = agent_id
        self.valuation: Valuation = valuation
        self.env: GameEnvironment = env
        self.belief: BeliefState = BeliefState(
            item_names=env.item_names,
            n_candidates=belief_candidates,
            seed=seed,
        )

    # ── strategy interface ────────────────────────────────────────────

    @abstractmethod
    def propose(self, history: History) -> Proposal:
        """
        Generate a Proposal for this round.

        Lesson 3: this implements the agent's *proposer strategy* — mapping
        the current information set (history + private valuation + belief)
        to an allocation offer.

        Parameters
        ----------
        history : all (round, proposer, proposal, response) events so far
        """

    @abstractmethod
    def respond(self, proposal: Proposal, history: History) -> Response:
        """
        React to an incoming Proposal.

        Returns one of: Accept, Reject, CounterProposal.

        Lesson 10: the responder's decision rule is their *responder strategy* —
        accept iff the offer exceeds their reservation value (at minimum the
        threat-point payoff; ideally their proportional fair share).

        Parameters
        ----------
        proposal : the incoming offer from the other agent
        history  : all events prior to this response
        """

    # ── belief observation (centralised update hook) ──────────────────

    def observe(self, opponent_self_bundle: Dict[str, int]) -> None:
        """
        Update this agent's belief about the opponent from an observed signal.

        This is the SINGLE entry point for Bayesian belief updates. The
        NegotiationProtocol calls it exactly once per round (on the responder,
        observing what the proposer kept for themselves). Centralising the
        update here — instead of scattering belief.update() calls inside
        respond() — guarantees:
          1. Each observation is counted exactly once (no double-counting /
             posterior overconfidence).
          2. Subclasses can override to keep auxiliary beliefs in sync (e.g.
             GeminiAgent also syncs its HeuristicAgent fallback's belief).

        Parameters
        ----------
        opponent_self_bundle : the bundle the opponent allocated to themselves
                               (a high-value signal about their private valuation)
        """
        self.belief.update(opponent_self_bundle)

    # ── shared helpers ────────────────────────────────────────────────

    @property
    def item_pool(self):
        """Convenience accessor for the game's total item pool."""
        return self.env.item_pool

    @property
    def own_fair_share(self) -> float:
        """
        The proportional fair share threshold for THIS agent.

        Lesson 10 – Proportionality:
            fair_share_i = U_i(all_items) / n_agents
        An agent should never accept less than this in a rational negotiation.
        """
        return self.valuation.max_possible_utility(self.item_pool) / len(self.env.agent_ids)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id})"
