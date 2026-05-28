"""
negotiation/messages.py
───────────────────────
Message types exchanged between agents during a negotiation round.

Theory basis (Lesson 10 – Negotiations):
  A negotiation protocol is a structured exchange of offers and counter-offers.
  Each message is a *move* in the extensive-form game tree (Lesson 6 – Decision Trees).
  Three moves are possible at each node:
    - Propose / CounterPropose : make a new offer
    - Accept                   : end the game, deal reached
    - Reject                   : end the game, no deal (threat point activated)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


# ─────────────────────────────────────────────
# Proposal
# ─────────────────────────────────────────────

@dataclass
class Proposal:
    """
    A complete allocation offer from one agent to the other.

    Attributes
    ----------
    proposer_id  : the agent making the offer
    bundles      : dict  agent_id → {item_name → count}
                   must cover BOTH agents and be feasible
    round_num    : the negotiation round this proposal belongs to
    reasoning    : optional free-text explanation (used by LLM agents)
    """
    proposer_id: str
    bundles: Dict[str, Dict[str, int]]
    round_num: int
    reasoning: Optional[str] = None

    def bundle_of(self, agent_id: str) -> Dict[str, int]:
        """Return the items allocated to agent_id under this proposal."""
        return dict(self.bundles.get(agent_id, {}))

    def __repr__(self) -> str:
        parts = " | ".join(f"{a}: {b}" for a, b in self.bundles.items())
        return f"Proposal(round={self.round_num}, by={self.proposer_id}, [{parts}])"


# ─────────────────────────────────────────────
# Response hierarchy
# ─────────────────────────────────────────────

@dataclass
class Accept:
    """
    The responder accepts the proposal — deal is reached.

    Game-theory note (Lesson 10):
        Acceptance is rational iff the proposed bundle gives the responder
        at least their reservation value (≥ threat-point + fair share).
    """
    responder_id: str
    accepted_proposal: Proposal
    reasoning: Optional[str] = None

    def __repr__(self) -> str:
        return f"Accept(by={self.responder_id}, round={self.accepted_proposal.round_num})"


@dataclass
class Reject:
    """
    The responder rejects with no counter-offer — negotiation terminates.

    Game-theory note (Lesson 10):
        Outright rejection activates the threat point: both agents receive d = (0, 0).
        A rational agent only rejects when no counter-offer can improve their position.
    """
    responder_id: str
    rejected_proposal: Proposal
    reasoning: Optional[str] = None

    def __repr__(self) -> str:
        return f"Reject(by={self.responder_id}, round={self.rejected_proposal.round_num})"


@dataclass
class CounterProposal:
    """
    The responder rejects the current offer and proposes a new allocation.

    Game-theory note (Lesson 6 – Decision Trees):
        A counter-proposal is a new decision node in the game tree.
        It carries the same structure as a Proposal but originates from
        the previous responder, who now takes the proposer role.
    """
    responder_id: str
    rejected_proposal: Proposal
    counter: Proposal
    reasoning: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"CounterProposal(by={self.responder_id}, "
            f"round={self.rejected_proposal.round_num} -> {self.counter})"
        )


# Type alias used throughout the codebase
Response = Accept | Reject | CounterProposal
