"""
test_core.py  —  smoke test for the pure-Python modules (zero API calls)
Run:  python test_core.py
"""
from game.environment import GameEnvironment
from game.allocation import Allocation, Valuation
from game.metrics import FairnessMetrics
from agents.beliefs import BeliefState


def test_environment():
    env = GameEnvironment.setup_random(seed=42)
    assert len(env.agent_ids) == 2
    assert sum(env.get_valuation("A").as_dict().values()) == 10
    assert sum(env.get_valuation("B").as_dict().values()) == 10
    print("  [PASS] GameEnvironment.setup_random")


def test_paper_example():
    env = GameEnvironment.setup_from_paper(
        val_A={"book": 4, "hat": 3, "ball": 3},
        val_B={"book": 2, "hat": 5, "ball": 3},
    )
    assert env.get_valuation("A").value_of("book") == 4
    assert env.get_valuation("B").value_of("hat") == 5
    print("  [PASS] GameEnvironment.setup_from_paper")


def test_utility():
    val = Valuation({"book": 4, "hat": 3, "ball": 3})
    assert val.utility_of({"book": 2, "hat": 1, "ball": 0}) == 4*2 + 3*1 + 3*0
    print("  [PASS] Valuation.utility_of")


def test_feasibility():
    alloc = Allocation(bundles={
        "A": {"book": 2, "hat": 1, "ball": 1},
        "B": {"book": 1, "hat": 1, "ball": 0},
    })
    env = GameEnvironment.setup_from_paper(
        val_A={"book": 4, "hat": 3, "ball": 3},
        val_B={"book": 2, "hat": 5, "ball": 3},
    )
    assert alloc.is_feasible(env.item_pool)
    print("  [PASS] Allocation.is_feasible")


def test_metrics():
    # A gets all books+balls, B gets all hats — easy split to test
    env = GameEnvironment.setup_from_paper(
        val_A={"book": 6, "hat": 0, "ball": 4},
        val_B={"book": 0, "hat": 10, "ball": 0},
    )
    alloc = Allocation(bundles={
        "A": {"book": 3, "hat": 0, "ball": 1},
        "B": {"book": 0, "hat": 2, "ball": 0},
    })
    report = FairnessMetrics.evaluate(alloc, env)
    # A gets utility 6*3+4*1=22, B gets 10*2=20
    assert report.utilities["A"] == 22.0
    assert report.utilities["B"] == 20.0
    assert report.above_threat       # both > 0
    print(f"  [PASS] FairnessMetrics.evaluate")
    print(f"         envy_free={report.envy_free}  proportional={report.proportional}  pareto={report.pareto_optimal}")


def test_beliefs():
    belief = BeliefState(item_names=["book", "hat", "ball"], n_candidates=20, seed=7)
    # Before any update: high entropy (uniform)
    h_before = belief.entropy()
    # Observe opponent keeping 2 books — should push belief toward book-loving profiles
    belief.update({"book": 3, "hat": 0, "ball": 0})
    h_after = belief.entropy()
    assert h_after <= h_before + 1e-9, "Entropy should not increase after an informative signal"
    eu = belief.calculate_expected_utility({"book": 2, "hat": 0, "ball": 0})
    assert eu > 0
    print(f"  [PASS] BeliefState update + expected utility (entropy: {h_before:.2f} -> {h_after:.2f})")


def test_envy_free_prediction():
    belief = BeliefState(item_names=["book", "hat", "ball"], n_candidates=20, seed=7)
    belief.update({"book": 3, "hat": 0, "ball": 0})  # opponent loves books
    # Offer opponent no books → they will likely envy a bundle with books
    would_envy = belief.would_opponent_envy(
        my_bundle={"book": 3, "hat": 1, "ball": 1},
        opponent_bundle={"book": 0, "hat": 1, "ball": 0},
    )
    assert would_envy, "Opponent who values books should envy a book-rich bundle"
    print("  [PASS] BeliefState.would_opponent_envy")


if __name__ == "__main__":
    print("Running core module smoke tests...\n")
    test_environment()
    test_paper_example()
    test_utility()
    test_feasibility()
    test_metrics()
    test_beliefs()
    test_envy_free_prediction()
    print("\nAll tests passed.")
