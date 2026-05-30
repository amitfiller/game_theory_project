"""
prompts/workflow_prompts.py
───────────────────────────
Prompt templates for baseline and workflow-guided LLM negotiation.

Paper basis (Section 5 / Algorithm 1):
  The paper compares two conditions:
    - Baseline : the LLM is given only the game state and told to negotiate.
    - Workflow  : the LLM is given a step-by-step reasoning scaffold that
                  explicitly mirrors Algorithm 1 (belief update, envy-free check,
                  EU maximisation, accept/counter decision).

  This is the core experimental manipulation: the workflow prompt is what makes
  the LLM behave as a rational game-theoretic agent (Lesson 12 connection).

Lesson alignment:
  Lesson  3 – Strategies: the workflow prompt enforces EU maximisation.
  Lesson 10 – Negotiations: explicit fair-share and envy-free reasoning steps.
  Lesson 12 – Game Theory & AI: structured prompting as a fix for LLM irrationality.
"""

from __future__ import annotations
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────
# PromptBuilder
# ─────────────────────────────────────────────

class PromptBuilder:
    """
    Builds system and turn prompts for both the baseline and workflow conditions.

    All methods are static — PromptBuilder is a namespace, not a stateful object.
    """

    # ── System prompts ────────────────────────────────────────────────

    @staticmethod
    def system_prompt(
        agent_id: str,
        valuation: Dict[str, int],
        item_pool: Dict[str, int],
        max_rounds: int,
        use_workflow: bool,
    ) -> str:
        """
        One-time system context injected at the start of the conversation.
        Tells the agent who it is, what items exist, and its private values.
        """
        val_str = ", ".join(f"{k}={v} pts" for k, v in valuation.items())
        pool_str = ", ".join(f"{k}×{v}" for k, v in item_pool.items())
        mode_tag = "[WORKFLOW MODE]" if use_workflow else "[BASELINE MODE]"

        base = f"""\
You are Agent {agent_id} in a two-player resource allocation negotiation. {mode_tag}

== GAME RULES ==
- Items in the pool: {pool_str}
- YOUR PRIVATE VALUATIONS (secret — never reveal the exact numbers): {val_str}
  (These sum to 10 — the convention from Lewis et al. 2017.)
- Negotiation lasts at most {max_rounds} rounds. If no deal is reached, BOTH players get 0.
- Each round: one player proposes an allocation, the other accepts, rejects, or counter-proposes.
- A deal is reached when one player accepts.

== GOAL ==
Maximise YOUR total utility (sum of value × count for items you receive).
"""

        if use_workflow:
            base += """
== WORKFLOW INSTRUCTIONS (follow these steps explicitly every turn) ==
When PROPOSING:
  Step 1 – BELIEF:  Based on the opponent's past proposals, estimate which items they value most.
  Step 2 – EU CALC: For each candidate allocation, estimate the opponent's expected utility.
  Step 3 – EF CHECK: Only consider allocations where the opponent's expected utility >= half of their total estimated value.
           This is the Envy-Freeness constraint: the opponent should not envy your bundle.
  Step 4 – MAXIMISE: Among EF-passing allocations, choose the one maximising YOUR utility.

When RESPONDING:
  Step 1 – UTILITY:    Calculate your utility for the bundle offered to you.
  Step 2 – FAIR SHARE: Your fair share = (sum of your values) / 2.
  Step 3 – DECIDE:
    - ACCEPT  if your utility >= your fair share.
    - COUNTER if your utility < your fair share (make a better proposal using Steps 1-4 above).
    - REJECT  only in the last round if no acceptable counter is possible.
"""
        else:
            base += """
== INSTRUCTIONS ==
Negotiate to get the best deal you can. You may accept, reject, or counter-propose each offer.
"""

        return base.strip()

    # ── Propose prompts ───────────────────────────────────────────────

    @staticmethod
    def propose_prompt(
        agent_id: str,
        opponent_id: str,
        item_pool: Dict[str, int],
        round_num: int,
        max_rounds: int,
        history_text: str,
        expected_opponent_valuation: Dict[str, float],
        belief_entropy: float,
        use_workflow: bool,
    ) -> str:
        """
        Turn prompt for generating a Proposal.

        Parameters
        ----------
        expected_opponent_valuation : posterior mean from BeliefState (used in workflow mode)
        belief_entropy              : current uncertainty level (shown for transparency)
        """
        pool_str = ", ".join(f"{k}: {v} available" for k, v in item_pool.items())
        ev_str = ", ".join(f"{k}≈{v:.1f}" for k, v in expected_opponent_valuation.items())

        prompt = f"== ROUND {round_num} of {max_rounds} — YOUR TURN TO PROPOSE ==\n\n"
        prompt += f"Items pool: {pool_str}\n\n"

        if history_text:
            prompt += f"=== NEGOTIATION HISTORY ===\n{history_text}\n\n"
        else:
            prompt += "This is the opening round — no history yet.\n\n"

        if use_workflow:
            prompt += (
                f"=== YOUR BELIEF STATE ===\n"
                f"Estimated opponent valuations (posterior mean): {ev_str}\n"
                f"Belief entropy: {belief_entropy:.2f} bits "
                f"({'high uncertainty' if belief_entropy > 3 else 'moderate' if belief_entropy > 1.5 else 'confident'})\n\n"
                f"Now follow the WORKFLOW INSTRUCTIONS from your system prompt.\n"
                f"Reason step by step before producing your output.\n\n"
            )
        else:
            prompt += "Make a proposal you think the opponent might accept.\n\n"

        prompt += (
            f"Output a JSON object with:\n"
            f"  my_bundle       : list of {{name, count}} for items YOU keep\n"
            f"  opponent_bundle : list of {{name, count}} for items given to {opponent_id}\n"
            f"  reasoning       : your explanation (required)\n\n"
            f"IMPORTANT: my_bundle + opponent_bundle must use ALL available items exactly.\n"
            f"Item counts must be non-negative integers."
        )
        return prompt

    # ── Respond prompts ───────────────────────────────────────────────

    @staticmethod
    def respond_prompt(
        agent_id: str,
        proposer_id: str,
        item_pool: Dict[str, int],
        proposal_text: str,
        my_utility_for_offer: float,
        my_fair_share: float,
        round_num: int,
        max_rounds: int,
        history_text: str,
        use_workflow: bool,
    ) -> str:
        """
        Turn prompt for responding to an incoming Proposal.

        Parameters
        ----------
        my_utility_for_offer : pre-computed utility so the LLM doesn't have to calculate it
        my_fair_share        : proportional fair share threshold for this agent
        """
        is_last_round = round_num >= max_rounds
        pool_str = ", ".join(f"{k}: {v} available" for k, v in item_pool.items())

        prompt = f"== ROUND {round_num} of {max_rounds} — RESPOND TO {proposer_id}'s PROPOSAL ==\n\n"
        prompt += f"Items pool: {pool_str}\n\n"

        if history_text:
            prompt += f"=== NEGOTIATION HISTORY ===\n{history_text}\n\n"

        prompt += f"=== INCOMING PROPOSAL from {proposer_id} ===\n{proposal_text}\n\n"

        if use_workflow:
            prompt += (
                f"=== WORKFLOW ANALYSIS ===\n"
                f"Your utility for the offered bundle: {my_utility_for_offer:.1f} pts\n"
                f"Your fair share threshold (= your_total_value / 2): {my_fair_share:.1f} pts\n"
                f"Offer {'MEETS' if my_utility_for_offer >= my_fair_share else 'DOES NOT MEET'} your fair share.\n\n"
                f"Follow Step 3 of your RESPOND WORKFLOW:\n"
            )
            if is_last_round:
                prompt += (
                    "  THIS IS THE LAST ROUND. If you reject or counter, BOTH players receive 0.\n"
                    "  Only accept if utility > 0; otherwise you are no worse off rejecting.\n\n"
                )
        else:
            prompt += f"Your utility for this offer: {my_utility_for_offer:.1f} pts\n\n"
            if is_last_round:
                prompt += "THIS IS THE LAST ROUND. If no deal is made, both players get 0.\n\n"

        prompt += (
            f"Output a JSON object with:\n"
            f"  action          : one of 'accept', 'reject', 'counter'\n"
            f"  my_bundle       : list of {{name, count}} for YOUR items (fill if action='counter', else [])\n"
            f"  opponent_bundle : list of {{name, count}} for {proposer_id} (fill if action='counter', else [])\n"
            f"  reasoning       : your explanation (required)\n\n"
            f"If action='counter': my_bundle + opponent_bundle must use ALL available items exactly."
        )
        return prompt

    # ── History formatter ─────────────────────────────────────────────

    @staticmethod
    def format_history(
        history: list,
        my_id: str,
    ) -> str:
        """Convert the raw history list into a human-readable string for the prompt."""
        if not history:
            return ""
        lines = []
        for round_num, proposer_id, proposal, response in history:
            lines.append(f"Round {round_num}:")
            role = "YOU proposed" if proposer_id == my_id else f"{proposer_id} proposed"
            my_bundle = proposal.bundle_of(my_id)
            opp_bundle = proposal.bundle_of(
                [a for a in proposal.bundles if a != my_id][0]
            )
            lines.append(
                f"  {role}: you get {my_bundle}, opponent gets {opp_bundle}"
            )
            from negotiation.messages import Accept, Reject, CounterProposal
            if isinstance(response, Accept):
                lines.append(f"  -> ACCEPTED by {response.responder_id}")
            elif isinstance(response, Reject):
                lines.append(f"  -> REJECTED by {response.responder_id}")
            elif isinstance(response, CounterProposal):
                counter_my = response.counter.bundle_of(my_id)
                counter_opp = response.counter.bundle_of(
                    [a for a in response.counter.bundles if a != my_id][0]
                )
                lines.append(
                    f"  -> COUNTER by {response.responder_id}: you get {counter_my}, "
                    f"opponent gets {counter_opp}"
                )
        return "\n".join(lines)
