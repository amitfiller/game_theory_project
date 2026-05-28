"""
allocation.py
─────────────
Core data structures for the resource-allocation negotiation game.

Theory basis (Lesson 2 – Game Definition & Utility):
  - Utility is a function mapping outcomes to real numbers.
  - Here: utility_i(allocation) = sum over items of value_i(item) * count_received(item)
  - Valuation vectors are private information (incomplete-information game).

Paper basis (Section 5, Algorithm 1):
  - Items: a fixed pool {book, hat, ball, …} each with a total count.
  - Valuations: each agent has a private integer vector that sums to 10 (normalisation
    convention from Lewis et al. 2017 "Deal or No Deal").
  - An Allocation assigns every item entirely to one agent (no partial splits).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# ─────────────────────────────────────────────
# Item
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Item:
    """
    A single type of resource in the negotiation pool.

    Attributes
    ----------
    name  : identifier used as a key throughout the system
    count : total units available to be split between the two agents
    """
    name: str
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"Item '{self.name}' must have count >= 1, got {self.count}.")


# ─────────────────────────────────────────────
# Valuation
# ─────────────────────────────────────────────

class Valuation:
    """
    An agent's private, integer-valued assessment of the items.

    Convention (Lewis et al. 2017 / paper Section 5):
        values must sum to exactly TOTAL_VALUE (default 10).
        This normalisation ensures a common unit for comparison.

    Lesson 2 connection:
        The Valuation is the agent's private 'type' in the incomplete-
        information game.  No opponent can observe it directly.
    """

    TOTAL_VALUE: int = 10

    def __init__(self, values: Dict[str, int]) -> None:
        """
        Parameters
        ----------
        values : dict mapping item name → non-negative integer value
        """
        if any(v < 0 for v in values.values()):
            raise ValueError("All valuation entries must be non-negative.")
        total = sum(values.values())
        if total != self.TOTAL_VALUE:
            raise ValueError(
                f"Valuation must sum to {self.TOTAL_VALUE}, got {total}."
            )
        self._values: Dict[str, int] = dict(values)

    # ── public interface ──────────────────────────────────────────────

    def value_of(self, item_name: str) -> int:
        """Return this agent's value for one unit of item_name."""
        return self._values.get(item_name, 0)

    def utility_of(self, bundle: Dict[str, int]) -> float:
        """
        Compute the agent's utility for a given bundle of items.

        Lesson 2 – Utility:
            U_i(bundle) = Σ  value_i(item) × count_received(item)

        Parameters
        ----------
        bundle : dict mapping item_name → count received by this agent
        """
        return float(sum(self._values.get(item, 0) * qty for item, qty in bundle.items()))

    def max_possible_utility(self, items: Dict[str, int]) -> float:
        """Upper bound on utility if the agent receives every item."""
        return self.utility_of(items)

    def as_dict(self) -> Dict[str, int]:
        """Return a copy of the raw valuation vector."""
        return dict(self._values)

    def __repr__(self) -> str:
        return f"Valuation({self._values})"


# ─────────────────────────────────────────────
# Allocation
# ─────────────────────────────────────────────

@dataclass
class Allocation:
    """
    A complete assignment of all items to the two agents.

    Attributes
    ----------
    bundles : dict  agent_id (str) → {item_name → count}
              e.g. {"A": {"book": 2, "hat": 0}, "B": {"book": 1, "hat": 2}}

    An allocation is *feasible* iff every item's counts sum to the pool total
    and no count is negative (Lesson 10 – Fair Division, feasibility condition).
    """

    bundles: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # ── feasibility ───────────────────────────────────────────────────

    def is_feasible(self, item_pool: Dict[str, int]) -> bool:
        """
        Check that the allocation distributes exactly the available items.

        Parameters
        ----------
        item_pool : dict  item_name → total count in the game
        """
        for item_name, total in item_pool.items():
            distributed = sum(
                self.bundles.get(agent, {}).get(item_name, 0)
                for agent in self.bundles
            )
            if distributed != total:
                return False
            for agent_bundle in self.bundles.values():
                if agent_bundle.get(item_name, 0) < 0:
                    return False
        return True

    # ── utility helpers ───────────────────────────────────────────────

    def utility_for(self, agent_id: str, valuation: Valuation) -> float:
        """
        Return agent agent_id's utility under *their own* valuation.

        Parameters
        ----------
        agent_id   : the agent whose bundle we evaluate
        valuation  : that agent's private Valuation object
        """
        bundle = self.bundles.get(agent_id, {})
        return valuation.utility_of(bundle)

    def bundle_of(self, agent_id: str) -> Dict[str, int]:
        """Return a copy of agent_id's bundle."""
        return dict(self.bundles.get(agent_id, {}))

    # ── display ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        parts = ", ".join(f"{a}={b}" for a, b in self.bundles.items())
        return f"Allocation({parts})"
