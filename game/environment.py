"""
environment.py
──────────────
GameEnvironment: the single source of truth for one negotiation instance.

Theory basis (Lesson 2 – Game Definition):
  A game is defined by: Players, Actions, Information, and Payoffs.
  The GameEnvironment encodes exactly these:
    - Players  : agent_ids ("A", "B")
    - Items    : the action space (what can be proposed)
    - Valuations: private payoff parameters (incomplete information)
    - max_rounds: the horizon (finite game tree – Lesson 6)

Separation of concerns:
  This module is the ONLY place that knows both agents' private valuations.
  Agents receive only their OWN valuation at game start — a clean implementation
  of the incomplete-information assumption from the paper (Section 5).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from game.allocation import Item, Valuation


class GameEnvironment:
    """
    Holds the full game setup: items, private valuations, and parameters.

    Usage
    -----
    >>> env = GameEnvironment.setup_random(seed=42)
    >>> print(env)

    Attributes
    ----------
    items        : list of Item objects (the resource pool)
    valuations   : dict  agent_id → Valuation  (PRIVATE — never share cross-agent)
    agent_ids    : ordered list of the two agent identifiers
    max_rounds   : negotiation horizon T (deal must be reached before round T+1)
    seed         : RNG seed for reproducibility
    """

    # Default item pool matching the Lewis et al. 2017 "Deal or No Deal" setup
    _DEFAULT_ITEMS: List[Tuple[str, int]] = [
        ("book", 3),
        ("hat",  2),
        ("ball", 1),
    ]

    def __init__(
        self,
        items: List[Item],
        valuations: Dict[str, Valuation],
        agent_ids: List[str],
        max_rounds: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        if len(agent_ids) != 2:
            raise ValueError("Exactly two agents are required.")
        if set(agent_ids) != set(valuations.keys()):
            raise ValueError("valuations must contain entries for both agent_ids.")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")

        self.items: List[Item] = items
        self.valuations: Dict[str, Valuation] = valuations
        self.agent_ids: List[str] = agent_ids
        self.max_rounds: int = max_rounds
        self.seed: Optional[int] = seed

    # ── item pool helpers ─────────────────────────────────────────────

    @property
    def item_pool(self) -> Dict[str, int]:
        """Return the total item counts as a plain dict {item_name: count}."""
        return {item.name: item.count for item in self.items}

    @property
    def item_names(self) -> List[str]:
        return [item.name for item in self.items]

    # ── agent-facing accessors ────────────────────────────────────────

    def get_valuation(self, agent_id: str) -> Valuation:
        """Return an agent's own private Valuation.  Called only by that agent."""
        if agent_id not in self.valuations:
            raise KeyError(f"Unknown agent_id: '{agent_id}'")
        return self.valuations[agent_id]

    def opponent_of(self, agent_id: str) -> str:
        """Return the other agent's id."""
        return [a for a in self.agent_ids if a != agent_id][0]

    # ── factory: random game ──────────────────────────────────────────

    @classmethod
    def setup_random(
        cls,
        items: Optional[List[Tuple[str, int]]] = None,
        agent_ids: Optional[List[str]] = None,
        max_rounds: int = 10,
        seed: Optional[int] = None,
    ) -> "GameEnvironment":
        """
        Create a game with randomly sampled private valuations.

        Valuations are sampled uniformly over all integer vectors that
        sum to Valuation.TOTAL_VALUE (= 10).  This is the same procedure
        the paper uses for its simulation experiments.

        Parameters
        ----------
        items      : list of (name, count) pairs; defaults to Lewis et al. pool
        agent_ids  : defaults to ["A", "B"]
        max_rounds : negotiation horizon T
        seed       : RNG seed for reproducibility
        """
        rng = random.Random(seed)
        items = items or cls._DEFAULT_ITEMS
        agent_ids = agent_ids or ["A", "B"]

        item_objects = [Item(name=n, count=c) for n, c in items]
        item_names = [n for n, _ in items]
        n_items = len(item_names)

        valuations: Dict[str, Valuation] = {}
        for agent in agent_ids:
            vals = cls._sample_valuation(rng, n_items)
            valuations[agent] = Valuation(dict(zip(item_names, vals)))

        return cls(
            items=item_objects,
            valuations=valuations,
            agent_ids=agent_ids,
            max_rounds=max_rounds,
            seed=seed,
        )

    @classmethod
    def setup_from_paper(
        cls,
        val_A: Dict[str, int],
        val_B: Dict[str, int],
        items: Optional[List[Tuple[str, int]]] = None,
        max_rounds: int = 10,
    ) -> "GameEnvironment":
        """
        Create a game with explicit valuations (for replicating paper examples).

        Example from Lewis et al. 2017:
            Agent A: book=4, hat=3, ball=3
            Agent B: book=2, hat=5, ball=3
        """
        items = items or cls._DEFAULT_ITEMS
        item_objects = [Item(name=n, count=c) for n, c in items]
        return cls(
            items=item_objects,
            valuations={"A": Valuation(val_A), "B": Valuation(val_B)},
            agent_ids=["A", "B"],
            max_rounds=max_rounds,
            seed=None,
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sample_valuation(rng: random.Random, n: int) -> List[int]:
        """
        Sample a random integer vector of length n that sums to TOTAL_VALUE.

        Method: stars-and-bars via sorted uniform breakpoints.
        Each vector is sampled with equal probability.
        """
        total = Valuation.TOTAL_VALUE
        # Generate n-1 random cut points in [1, total-1], sort them,
        # then compute gaps.  This gives a uniform distribution over
        # all non-negative integer compositions that sum to total.
        cuts = sorted(rng.sample(range(1, total), n - 1))
        boundaries = [0] + cuts + [total]
        return [boundaries[i + 1] - boundaries[i] for i in range(n)]

    # ── display ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        pool = ", ".join(f"{i.name}×{i.count}" for i in self.items)
        agents = " | ".join(
            f"{a}: {self.valuations[a]}" for a in self.agent_ids
        )
        return (
            f"GameEnvironment(\n"
            f"  pool      = [{pool}]\n"
            f"  rounds    = {self.max_rounds}\n"
            f"  {agents}\n"
            f")"
        )
