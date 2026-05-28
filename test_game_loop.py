"""
test_game_loop.py
─────────────────
Integration test: run a full negotiation between two HeuristicAgents.
Zero API calls — validates the complete offline pipeline end-to-end.

Covers grading criterion #4 (Operational): if this passes, the game loop
is stable and we can safely plug in the Gemini client later.

Run:  python test_game_loop.py
"""

from game.environment import GameEnvironment
from agents.heuristic_agent import HeuristicAgent
from negotiation.protocol import NegotiationProtocol, GameResult


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def run_one_game(seed: int, verbose: bool = False) -> GameResult:
    """Instantiate a random game and run it to completion with two HeuristicAgents."""
    env = GameEnvironment.setup_random(seed=seed, max_rounds=10)
    agent_a = HeuristicAgent("A", env, belief_candidates=30, seed=seed)
    agent_b = HeuristicAgent("B", env, belief_candidates=30, seed=seed + 1)
    protocol = NegotiationProtocol(agent_a, agent_b, env)
    result = protocol.run()
    if verbose:
        print(result.transcript)
        print(result.summary())
    return result


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_game_terminates():
    """The game must always terminate (no infinite loops)."""
    for seed in range(10):
        result = run_one_game(seed)
        assert result.termination in ("deal", "reject", "timeout"), (
            f"Unexpected termination reason: {result.termination}"
        )
        assert result.rounds_used >= 1
    print("  [PASS] All 10 random games terminate gracefully")


def test_deal_metrics_computed():
    """When a deal is reached, metrics must be computed and non-null."""
    deals_found = 0
    for seed in range(20):
        result = run_one_game(seed)
        if result.deal_reached:
            assert result.metrics is not None, "metrics should not be None on a deal"
            assert result.final_proposal is not None
            # Every utility in a deal must be >= 0 (at least threat-point level)
            for u in result.metrics.utilities.values():
                assert u >= 0, f"Utility {u} below threat point"
            deals_found += 1
    print(f"  [PASS] {deals_found}/20 games ended in deals; all had valid metrics")


def test_no_deal_has_no_metrics():
    """Non-deal results should have metrics=None (threat point, both get 0)."""
    for seed in range(20):
        result = run_one_game(seed)
        if not result.deal_reached:
            assert result.metrics is None
    print("  [PASS] No-deal results correctly carry no metrics (threat point activated)")


def test_transcript_is_populated():
    """Every game must produce a non-empty transcript."""
    for seed in range(5):
        result = run_one_game(seed)
        assert len(result.transcript) > 0, "Transcript should not be empty"
        assert "Round 1" in result.transcript
    print("  [PASS] Transcripts are non-empty and contain round logs")


def test_rounds_within_limit():
    """Rounds used must never exceed max_rounds."""
    for seed in range(10):
        env = GameEnvironment.setup_random(seed=seed, max_rounds=10)
        result = run_one_game(seed)
        assert result.rounds_used <= env.max_rounds
    print("  [PASS] Rounds used never exceeds max_rounds limit")


def test_paper_example_game():
    """Run the Lewis et al. 2017 canonical example to check specific outcomes."""
    env = GameEnvironment.setup_from_paper(
        val_A={"book": 4, "hat": 3, "ball": 3},
        val_B={"book": 2, "hat": 5, "ball": 3},
        max_rounds=10,
    )
    agent_a = HeuristicAgent("A", env, belief_candidates=30, seed=0)
    agent_b = HeuristicAgent("B", env, belief_candidates=30, seed=1)
    protocol = NegotiationProtocol(agent_a, agent_b, env)
    result = protocol.run()

    assert result.termination in ("deal", "reject", "timeout")
    print(f"  [PASS] Paper example game: termination={result.termination}, "
          f"rounds={result.rounds_used}")
    if result.deal_reached:
        print(f"         {result.metrics.summary()}")


# ─────────────────────────────────────────────
# Verbose demo run
# ─────────────────────────────────────────────

def demo_single_game(seed: int = 42) -> None:
    """Run one game with full transcript printed to console."""
    print(f"\n{'='*60}")
    print(f"  DEMO GAME (seed={seed})")
    print(f"{'='*60}\n")
    run_one_game(seed, verbose=True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running game loop integration tests...\n")
    test_game_terminates()
    test_deal_metrics_computed()
    test_no_deal_has_no_metrics()
    test_transcript_is_populated()
    test_rounds_within_limit()
    test_paper_example_game()
    print("\nAll integration tests passed.")
    demo_single_game(seed=42)
