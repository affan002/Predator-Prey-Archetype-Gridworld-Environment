# test_iql_train2v2.py
import numpy as np
import types
import pytest

import iql_train2v2 as iql

# ------------- 1. Agent and Environment Setup -----------------


def test_make_agents_instantiation():
    """Test Case 1.1: Verify correct agent creation and naming."""
    agents = iql.make_agents(num_predators=2, num_preys=2)

    assert len(agents) == 4
    names = [ag.agent_name for ag in agents]
    types_ = [ag.agent_type for ag in agents]

    # Ensure expected names and types appear exactly once
    assert set(names) == {"prey_1", "prey_2", "predator_1", "predator_2"}
    assert types_.count("prey") == 2
    assert types_.count("predator") == 2
    print("Success")


def test_make_env_and_meta_state_space():
    """Test Case 1.2: Ensure n_states = (grid_size^2)^n_agents."""

    agents = iql.make_agents(num_predators=2, num_preys=2)
    grid_size = 2
    env, n_states, n_actions = iql.make_env_and_meta(agents, grid_size, seed=0)

    expected_n_states = (grid_size * grid_size) ** len(agents)
    assert n_states == expected_n_states
    assert n_actions == 5
    assert type(env).__name__ == "GridWorldEnv"
    print("Success")


# ------------- 2. Joint State Indexing -----------------


def test_joint_state_index_determinism():
    """Test Case 2.1: Verify deterministic index value."""
    grid_size = 5
    positions = [(0, 0), (1, 1), (4, 4)]
    # Manual ground truth: 0*(25^2)+6*(25)+24 = 174
    n_cells = grid_size * grid_size
    ground_truth = 174
    assert iql.joint_state_index(positions, grid_size) == ground_truth
    print("Success")


# ------------- 3. Epsilon-Greedy Action Selection -----------------


def test_epsilon_greedy_pure_exploration():
    """Test Case 3.1: ε=1.0 should always sample randomly."""
    q_row = np.array([0.1, 0.5, 0.2, 0.0, 1.0])
    n_actions = len(q_row)
    rng = np.random.default_rng(42)
    runs = 1000

    results = [
        iql.epsilon_greedy_action(q_row, n_actions, rng, eps=1.0) for _ in range(runs)
    ]
    # should only contain valid actions
    assert all(0 <= r < n_actions for r in results)
    # Should contain multiple distinct actions (randomness check)
    assert len(set(results)) > 1
    # Basic uniformity check (not strict)
    action_counts = np.bincount(results, minlength=n_actions)
    expected_count = runs / n_actions
    assert all(
        abs(count - expected_count) < 0.5 * expected_count for count in action_counts
    )
    print("Success")


def test_epsilon_greedy_pure_exploitation():
    """Test Case 3.2: ε=0.0 returns argmax."""
    q_row = np.array([1.0, 5.0, 2.0])
    rng = np.random.default_rng(123)
    for _ in range(20):  # run a few times to ensure consistency
        action = iql.epsilon_greedy_action(q_row, len(q_row), rng, eps=0.0)
        assert action == 1  # index of max value
    print("Success")


def test_epsilon_greedy_exploitation_with_ties():
    """Test Case 3.3: Multiple maxima—return one of optimal indices."""
    q_row = np.array([5.0, 5.0, 1.0])
    rng = np.random.default_rng(999)
    valid = {0, 1}
    # Run several times to see both possibilities
    results = {iql.epsilon_greedy_action(q_row, 3, rng, eps=0.0) for _ in range(20)}
    assert results.issubset(valid)
    print("Success")


# ------------- 4. Q-Table Initialization -----------------


def test_init_q_tables_shape_and_values():
    """Test Case 4.1: Verify structure and zero-initialization."""
    agent_names = ["a1", "a2", "a3"]
    n_states, n_actions = 10, 4
    Qs = iql.init_q_tables(agent_names, n_states, n_actions)

    # Dict keys match agents
    assert set(Qs.keys()) == set(agent_names)

    # Each Q-table has correct shape and zeros
    for name in agent_names:
        Q = Qs[name]
        assert Q.shape == (n_states, n_actions)
        assert np.allclose(Q, 0.0)
        assert Q.dtype == np.float32
    print("Success")


def test_init_q_tables_independent_tables():
    """Test Case 4.2: Ensure each Q-table is independent (no shared memory)."""
    agent_names = ["x", "y"]
    n_states, n_actions = 3, 2
    Qs = iql.init_q_tables(agent_names, n_states, n_actions)

    Qs["x"][0, 0] = 99.0
    assert Qs["y"][0, 0] == 0.0  # Modifying one should not affect the other
    print("Success")


if __name__ == "__main__":
    test_make_agents_instantiation()
    test_make_env_and_meta_state_space()
    test_joint_state_index_determinism()
    test_epsilon_greedy_pure_exploration()
    test_epsilon_greedy_pure_exploitation()
    test_epsilon_greedy_exploitation_with_ties()
    test_init_q_tables_shape_and_values()
    test_init_q_tables_independent_tables()
    print("All tests passed.")
