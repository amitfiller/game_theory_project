"""
beliefs.py
──────────
Bayesian belief state over the opponent's private valuations.

Theory basis (Lesson 3 – Strategies & Decision Making):
  In an incomplete-information game, each agent has a *type* (their private
  valuation vector) that is unknown to the opponent.  Rational agents maintain
  a probability distribution (belief) over the set of possible opponent types
  and update it with Bayes' Rule every time they observe new information.

  This module implements exactly that:
    1. Prior  : a uniform distribution over a finite set of candidate profiles.
    2. Update : Bayes' Rule applied to the observed proposal (signal).
    3. EU calc: Expected Utility Maximization over the posterior.

Bayes' Rule (Lesson 3):
    P(type | signal) ∝ P(signal | type) × P(type)

  Here:
    - type   = a candidate valuation profile for the opponent
    - signal = the allocation the opponent just proposed for themselves
    - P(signal | type): likelihood that a rational agent with `type` would
                        propose this allocation (proportional to utility gained)
    - P(type): current prior/posterior weight for this profile

Paper basis (Section 5 / Algorithm 1):
  "The agent updates its belief about the opponent's valuation based on
   the items the opponent kept in their proposal."
  Items the opponent allocates to themselves are high-value signals.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from game.allocation import Valuation


# ─────────────────────────────────────────────
# BeliefState
# ─────────────────────────────────────────────

class BeliefState:
    """
    A probability distribution over a discrete set of opponent valuation profiles.

    The distribution is represented as a list of (profile, weight) pairs where
    the weights are non-negative and sum to 1 (a proper probability distribution).

    Attributes
    ----------
    item_names   : names of all items in the game (determines profile shape)
    n_candidates : number of candidate profiles to sample at init
    profiles     : list of Valuation objects (the support of the distribution)
    weights      : parallel list of floats (probability mass for each profile)
    """

    def __init__(
        self,
        item_names: List[str],
        n_candidates: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialise with a uniform prior over n_candidates randomly sampled profiles.

        Lesson 3 – Prior:
            When no information is available, we assume a uniform prior.
            Every candidate valuation profile is equally likely.

        Parameters
        ----------
        item_names   : list of item names (e.g. ["book", "hat", "ball"])
        n_candidates : size of the discrete support (more → smoother posterior)
        seed         : RNG seed (for reproducibility across runs)
        """
        self.item_names: List[str] = item_names
        self._rng = random.Random(seed)

        self.profiles: List[Valuation] = self._sample_profiles(n_candidates)
        # Uniform prior: every profile starts with equal weight
        n = len(self.profiles)
        self.weights: List[float] = [1.0 / n] * n

    # ── Bayesian Update ───────────────────────────────────────────────

    def update(self, observed_opponent_bundle: Dict[str, int]) -> None:
        """
        Apply Bayes' Rule given the opponent's observed self-allocation.

        Lesson 3 – Bayesian Update:
            posterior(profile) ∝ likelihood(observation | profile) × prior(profile)

        Likelihood model:
            A rational opponent with valuation profile v would prefer to keep
            items they value highly.  We model this with a softmax over utility:

            P(observation | profile) ∝ exp( β × U(observation | profile) )

            where β (inverse temperature) controls how "peaked" the likelihood is.
            High β → the agent is assumed to be very rational (greedily optimal).
            β = 1.0 is a reasonable default.

        Paper connection (Section 5):
            "Items the opponent kept in their proposal provide a signal about
             their private valuation."

        Parameters
        ----------
        observed_opponent_bundle : dict  item_name → count the opponent proposed
                                   to keep for themselves
        """
        BETA = 1.0  # rationality coefficient

        # Compute un-normalised posterior weights
        new_weights: List[float] = []
        for profile, prior_weight in zip(self.profiles, self.weights):
            # Utility this profile assigns to the observed bundle
            utility = profile.utility_of(observed_opponent_bundle)
            # Likelihood: proportional to exp(β × utility)
            likelihood = math.exp(BETA * utility)
            new_weights.append(likelihood * prior_weight)

        # Normalise so weights sum to 1
        total = sum(new_weights)
        if total == 0:
            # Degenerate case: reset to uniform (safety net)
            n = len(self.profiles)
            self.weights = [1.0 / n] * n
        else:
            self.weights = [w / total for w in new_weights]

    # ── Expected Utility ──────────────────────────────────────────────

    def calculate_expected_utility(
        self, opponent_bundle: Dict[str, int]
    ) -> float:
        """
        Compute the expected utility the opponent would get from a given bundle,
        averaged over the posterior distribution of their type.

        Lesson 3 – Expected Utility Maximisation:
            EU_i(action) = Σ_{type ∈ support} P(type) × U_i(action | type)

        This is the key quantity used when the proposer checks envy-freeness:
        "Would the opponent (in expectation) prefer my bundle or their bundle?"

        Parameters
        ----------
        opponent_bundle : the bundle we want to evaluate (items for the opponent)

        Returns
        -------
        float : E[U_opponent(opponent_bundle)] under the current posterior
        """
        expected_u = 0.0
        for profile, weight in zip(self.profiles, self.weights):
            expected_u += weight * profile.utility_of(opponent_bundle)
        return expected_u

    # ── Envy-Free Proposal Helper ────────────────────────────────────

    def would_opponent_envy(
        self,
        my_bundle: Dict[str, int],
        opponent_bundle: Dict[str, int],
    ) -> bool:
        """
        Predict whether the opponent would envy the proposer's bundle.

        An allocation is envy-free for the opponent iff:
            E[U_opponent(their_bundle)] >= E[U_opponent(my_bundle)]

        Paper / Algorithm 1:
            The proposer uses this check to filter candidate proposals —
            only proposals where the opponent would NOT envy are submitted.

        Parameters
        ----------
        my_bundle       : items the proposer would keep
        opponent_bundle : items offered to the opponent

        Returns
        -------
        bool : True if the opponent would (in expectation) envy the proposer
        """
        eu_theirs = self.calculate_expected_utility(opponent_bundle)
        eu_mine = self.calculate_expected_utility(my_bundle)
        return eu_mine > eu_theirs  # opponent envies if proposer's EU > their own

    # ── Posterior Summary ─────────────────────────────────────────────

    def expected_valuation(self) -> Dict[str, float]:
        """
        Return the posterior mean valuation vector.

        This is a single "best guess" of the opponent's type, useful for
        logging and for building intuitions in the presentation.

        Returns
        -------
        dict  item_name → posterior mean value
        """
        mean: Dict[str, float] = {item: 0.0 for item in self.item_names}
        for profile, weight in zip(self.profiles, self.weights):
            for item in self.item_names:
                mean[item] += weight * profile.value_of(item)
        return mean

    def most_likely_profile(self) -> Tuple[Valuation, float]:
        """Return the MAP (maximum a posteriori) profile and its probability."""
        best_idx = max(range(len(self.weights)), key=lambda i: self.weights[i])
        return self.profiles[best_idx], self.weights[best_idx]

    def entropy(self) -> float:
        """
        Shannon entropy of the current belief distribution (in bits).

        High entropy → high uncertainty about the opponent's type.
        Low entropy  → confident belief concentrated on few profiles.
        Entropy decreases as we accumulate observations (good Bayesian signal).
        """
        h = 0.0
        for w in self.weights:
            if w > 1e-12:
                h -= w * math.log2(w)
        return h

    # ── Profile Sampling ─────────────────────────────────────────────

    def _sample_profiles(self, n: int) -> List[Valuation]:
        """
        Sample n distinct valuation profiles that sum to TOTAL_VALUE (= 10).

        Uses the stars-and-bars method (same as GameEnvironment._sample_valuation)
        to ensure uniform coverage of the type space.
        """
        total = Valuation.TOTAL_VALUE
        k = len(self.item_names)
        profiles: List[Valuation] = []
        seen: set = set()

        attempts = 0
        while len(profiles) < n and attempts < n * 100:
            attempts += 1
            cuts = sorted(self._rng.sample(range(1, total), k - 1))
            boundaries = [0] + cuts + [total]
            vals = tuple(boundaries[i + 1] - boundaries[i] for i in range(k))
            if vals not in seen:
                seen.add(vals)
                profiles.append(Valuation(dict(zip(self.item_names, vals))))

        if len(profiles) < n:
            # Fallback: duplicate profiles if the space is too small
            while len(profiles) < n:
                profiles.append(profiles[len(profiles) % len(profiles)])

        return profiles

    # ── Display ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        mean = self.expected_valuation()
        map_profile, map_prob = self.most_likely_profile()
        return (
            f"BeliefState(\n"
            f"  support_size = {len(self.profiles)}\n"
            f"  entropy      = {self.entropy():.3f} bits\n"
            f"  mean_belief  = {mean}\n"
            f"  MAP_profile  = {map_profile}  (p={map_prob:.3f})\n"
            f")"
        )
