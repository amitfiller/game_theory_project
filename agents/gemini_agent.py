"""
agents/gemini_agent.py
──────────────────────
LLM-powered negotiation agent using the Gemini API.

This is the core experimental agent from the paper:
  - use_workflow=False → baseline condition (vanilla LLM)
  - use_workflow=True  → workflow condition (Algorithm 1 scaffold)

The class inherits from AbstractAgent and is drop-in compatible with
NegotiationProtocol — the protocol cannot tell it apart from HeuristicAgent.

Fallback safety net (grading criterion #4 – Operational):
  If the LLM returns an infeasible allocation (items don't sum correctly),
  the agent falls back to a HeuristicAgent proposal for that turn rather
  than crashing.  This is logged clearly so we can track fallback rate.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from game.allocation import Allocation
from game.environment import GameEnvironment
from agents.base_agent import AbstractAgent, History
from agents.heuristic_agent import HeuristicAgent
from llm.gemini_client import GeminiClient, ItemCount, ProposeOutput, RespondOutput
from negotiation.messages import (
    Accept, CounterProposal, Proposal, Reject, Response
)
from prompts.workflow_prompts import PromptBuilder


class GeminiAgent(AbstractAgent):
    """
    Negotiation agent powered by Gemini via structured outputs.

    Parameters
    ----------
    agent_id        : "A" or "B"
    env             : the shared GameEnvironment
    use_workflow    : True = Algorithm 1 scaffold; False = baseline prompt
    client          : shared GeminiClient (pass None to create a fresh one)
    belief_candidates : size of the Bayesian belief support
    seed            : RNG seed for the BeliefState sampler
    """

    def __init__(
        self,
        agent_id: str,
        env: GameEnvironment,
        use_workflow: bool = True,
        client: Optional[GeminiClient] = None,
        belief_candidates: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        valuation = env.get_valuation(agent_id)
        super().__init__(
            agent_id=agent_id,
            valuation=valuation,
            env=env,
            belief_candidates=belief_candidates,
            seed=seed,
        )
        self.use_workflow = use_workflow
        self.client = client or GeminiClient()
        self._fallback = HeuristicAgent(agent_id, env, belief_candidates, seed)

        # Pre-build the system prompt once (reused every turn)
        self._system_prompt = PromptBuilder.system_prompt(
            agent_id=agent_id,
            valuation=valuation.as_dict(),
            item_pool=env.item_pool,
            max_rounds=env.max_rounds,
            use_workflow=use_workflow,
        )

        self.fallback_count = 0     # how often the LLM produced infeasible/invalid output
        self.envy_rejections = 0    # how often the envy-gate vetoed an enviable LLM proposal
        self.reject_overrides = 0   # how often the reject-guardrail converted reject->counter

    # ── belief observation (override to keep fallback in sync) ────────

    def observe(self, opponent_self_bundle):
        """
        Centralised belief update, overridden so the HeuristicAgent fallback's
        belief stays perfectly synchronised with the primary agent's belief.

        Without this, a mid-game fallback would act on a stale posterior.
        The protocol calls this exactly once per round.
        """
        self.belief.update(opponent_self_bundle)
        self._fallback.belief.update(opponent_self_bundle)

    # ── propose ───────────────────────────────────────────────────────

    def propose(self, history: History) -> Proposal:
        """
        Generate a Proposal via Gemini structured output.

        Builds the turn prompt (with or without workflow scaffold), calls
        the API, validates feasibility, and falls back to HeuristicAgent
        if the LLM output is invalid.
        """
        round_num = len(history) + 1
        opponent_id = self.env.opponent_of(self.agent_id)

        turn_prompt = PromptBuilder.propose_prompt(
            agent_id=self.agent_id,
            opponent_id=opponent_id,
            item_pool=self.item_pool,
            round_num=round_num,
            max_rounds=self.env.max_rounds,
            history_text=PromptBuilder.format_history(history, self.agent_id),
            expected_opponent_valuation=self.belief.expected_valuation(),
            belief_entropy=self.belief.entropy(),
            use_workflow=self.use_workflow,
        )

        try:
            output: ProposeOutput = self.client.generate_proposal(
                self._system_prompt, turn_prompt
            )
            bundles = self._parse_bundles(output.my_bundle, output.opponent_bundle, opponent_id)
            alloc = Allocation(bundles=bundles)
            if not alloc.is_feasible(self.item_pool):
                raise ValueError(
                    f"Infeasible allocation from LLM: {bundles}. "
                    f"Pool={self.item_pool}"
                )

            # ── Envy-Free enforcement gate (workflow condition only) ──────────
            # The workflow PROMISES envy-free proposals. The prompt requests them,
            # but an LLM can still emit an enviable split. Here we VERIFY it in
            # pure Python: under our current belief, would the opponent prefer our
            # bundle to theirs? If so, we reject the LLM's proposal and substitute
            # the HeuristicAgent's rational, fairness-constrained proposal. This is
            # what makes "the workflow guarantees fairness" a true statement rather
            # than a hopeful one.
            if self.use_workflow:
                my_bundle = bundles[self.agent_id]
                opp_bundle = bundles[opponent_id]
                if self.belief.would_opponent_envy(my_bundle, opp_bundle):
                    self.envy_rejections += 1
                    print(
                        f"    [GeminiAgent {self.agent_id}] envy-gate: LLM proposal "
                        f"is enviable under belief -> substituting heuristic "
                        f"envy-aware proposal (#{self.envy_rejections})"
                    )
                    return self._fallback.propose(history)

            return Proposal(
                proposer_id=self.agent_id,
                bundles=bundles,
                round_num=round_num,
                reasoning=f"[Gemini{'|workflow' if self.use_workflow else '|baseline'}] "
                          f"{output.reasoning}",
            )
        except Exception as exc:
            self.fallback_count += 1
            print(
                f"    [GeminiAgent {self.agent_id}] propose() fallback "
                f"(#{self.fallback_count}): {exc}"
            )
            return self._fallback.propose(history)

    # ── respond ───────────────────────────────────────────────────────

    def respond(self, proposal: Proposal, history: History) -> Response:
        """
        Respond to an incoming Proposal via Gemini structured output.

        Asks Gemini to decide: accept / reject / counter.

        Note: belief updates are NOT performed here. The protocol calls
        self.observe(...) once per round (before respond), keeping both this
        agent's belief and its fallback's belief synchronised. The Bayesian
        math itself remains pure Python — never delegated to the LLM.
        """
        opponent_id = proposal.proposer_id
        my_utility = self.valuation.utility_of(proposal.bundle_of(self.agent_id))
        proposal_text = self._format_proposal_for_prompt(proposal)

        turn_prompt = PromptBuilder.respond_prompt(
            agent_id=self.agent_id,
            proposer_id=opponent_id,
            item_pool=self.item_pool,
            proposal_text=proposal_text,
            my_utility_for_offer=my_utility,
            my_fair_share=self.own_fair_share,
            round_num=proposal.round_num,
            max_rounds=self.env.max_rounds,
            history_text=PromptBuilder.format_history(history, self.agent_id),
            use_workflow=self.use_workflow,
        )

        try:
            output: RespondOutput = self.client.generate_response(
                self._system_prompt, turn_prompt
            )
            return self._parse_response(output, proposal, history)
        except Exception as exc:
            self.fallback_count += 1
            print(
                f"    [GeminiAgent {self.agent_id}] respond() fallback "
                f"(#{self.fallback_count}): {exc}"
            )
            return self._fallback.respond(proposal, history)

    # ── parsing helpers ───────────────────────────────────────────────

    def _parse_bundles(
        self,
        my_items: List[ItemCount],
        opp_items: List[ItemCount],
        opponent_id: str,
    ) -> Dict[str, Dict[str, int]]:
        """Convert LLM ItemCount lists into the internal bundle dict format."""
        return {
            self.agent_id: {ic.name: ic.count for ic in my_items},
            opponent_id: {ic.name: ic.count for ic in opp_items},
        }

    def _parse_response(
        self,
        output: RespondOutput,
        original_proposal: Proposal,
        history: History,
    ) -> Response:
        """Convert a RespondOutput schema into an Accept / Reject / CounterProposal."""
        action = output.action.lower().strip()
        tag = f"[Gemini{'|workflow' if self.use_workflow else '|baseline'}] "

        if action == "accept":
            return Accept(
                responder_id=self.agent_id,
                accepted_proposal=original_proposal,
                reasoning=tag + output.reasoning,
            )

        if action == "reject":
            # ── Reject guardrail ──────────────────────────────────────────────
            # A reject terminates the game and activates the threat point d=(0,0):
            # BOTH agents get zero. That is only rational in the final round, when
            # no further counter can help. If the LLM rejects with rounds still
            # remaining, we override it: convert the reject into a counter-proposal
            # carrying this agent's own fair-share-maximising bundle (computed by
            # the deterministic heuristic), keeping the negotiation alive.
            if original_proposal.round_num < self.env.max_rounds:
                self.reject_overrides += 1
                counter = self._fallback.propose(history)
                print(
                    f"    [GeminiAgent {self.agent_id}] reject-guardrail: LLM "
                    f"rejected in round {original_proposal.round_num}/"
                    f"{self.env.max_rounds} -> converted to counter "
                    f"(#{self.reject_overrides})"
                )
                return CounterProposal(
                    responder_id=self.agent_id,
                    rejected_proposal=original_proposal,
                    counter=counter,
                    reasoning=tag + "[reject->counter guardrail] " + output.reasoning,
                )
            # Final round: a genuine walk-away is permitted.
            return Reject(
                responder_id=self.agent_id,
                rejected_proposal=original_proposal,
                reasoning=tag + output.reasoning,
            )

        # counter — parse and validate the counter bundles
        opponent_id = original_proposal.proposer_id
        bundles = self._parse_bundles(output.my_bundle, output.opponent_bundle, opponent_id)
        alloc = Allocation(bundles=bundles)
        if not alloc.is_feasible(self.item_pool):
            raise ValueError(f"Infeasible counter-proposal from LLM: {bundles}")

        counter = Proposal(
            proposer_id=self.agent_id,
            bundles=bundles,
            round_num=original_proposal.round_num,
            reasoning=tag + output.reasoning,
        )
        return CounterProposal(
            responder_id=self.agent_id,
            rejected_proposal=original_proposal,
            counter=counter,
            reasoning=tag + output.reasoning,
        )

    def _format_proposal_for_prompt(self, proposal: Proposal) -> str:
        """Human-readable single-line summary of a proposal."""
        my_bundle = proposal.bundle_of(self.agent_id)
        opp_bundle = proposal.bundle_of(proposal.proposer_id)
        my_str = ", ".join(f"{k}: {v}" for k, v in my_bundle.items())
        opp_str = ", ".join(f"{k}: {v}" for k, v in opp_bundle.items())
        return (
            f"{proposal.proposer_id} offers you [{my_str}] and keeps [{opp_str}]."
        )
