from typing import List, Dict
import argparse
import time

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent
from multi_agent_package.helpers.helper import print_action, print_mgp_info


def build_agents() -> List[Agent]:
    """Create and return a list of Agent instances for the demo.

    Adjust or replace this factory to configure a different scenario.
    """
    # Note: agent_team here is numeric subteam id (int) for clarity
    agent101 = Agent("prey", 1, "PY101_Tom")
    agent102 = Agent("prey", 2, "PY102_Garfield")

    agent201 = Agent("prey", 3, "PY201_Jerry")
    agent202 = Agent("predator", 1, "PD202_Stuart")

    # You can add more agents here
    agents = [agent101, agent102, agent201, agent202]
    return agents


def assign_total_subteams(agents: List[Agent]) -> None:
    """Auto-detect total subteams per base type and set `agent.total_subteams`.

    This makes color/shape spacing adaptive so that contrast is maximized when
    there are few subteams and reduced when there are many.
    """
    # Compute max sub_id seen for each base type
    counts: Dict[str, int] = {}
    for ag in agents:
        try:
            base, sub_id = ag._parse_team()
        except Exception:
            base = ag.agent_type
            sub_id = int(ag.agent_team) if isinstance(ag.agent_team, int) else 1
        counts[base] = max(counts.get(base, 0), int(sub_id))

    # Apply totals back to agents
    for ag in agents:
        base, _ = ag._parse_team()
        ag.total_subteams = counts.get(base, 1)


def run_simulation(
    agents: List[Agent],
    size: int = 10,
    perc_num_obstacle: float = 10.0,
    render_mode: str = "human",
    steps: int = 100,
    seed: int = 0,
) -> None:
    """Run a simple simulation loop.

    Parameters
    ----------
    agents : list[Agent]
        Agents participating in the environment.
    size : int
        Grid size (size x size).
    perc_num_obstacle : float
        Percent of grid occupied by obstacles.
    render_mode : str
        'human' for pygame window or 'rgb_array' to get frames returned.
    steps : int
        Number of simulation steps to run.
    seed : int
        RNG seed for reproducibility.
    """
    assign_total_subteams(agents)

    env = GridWorldEnv(agents=agents, render_mode=render_mode, size=size, perc_num_obstacle=perc_num_obstacle)

    try:
        current_state, info = env.reset()

        action: Dict[str, int] = {}

        for t in range(steps):
            # Choose a random action for every agent (replace with a policy later)
            for ag in agents:
                action[ag.agent_name] = env.action_space.sample()

            mgp_tuple = env.step(action)

            # Use helper to print/pretty-print step information (if available)
            try:
                print_mgp_info(mgp_tuple, t, current_state, action)
            except Exception:
                # Fallback simple print for robustness
                print(f"Step {t}: actions={action}, reward={mgp_tuple.get('reward')}")

            if mgp_tuple.get("terminated"):
                print("Episode terminated at step", t)
                break

            current_state = mgp_tuple["obs"]
            # slight pause so human render is visible at a readable speed
            if render_mode == "human":
                time.sleep(0.01)

    finally:
        # Ensure resources are freed
        env.close()
        print("Environment closed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GridWorldEnv demo")
    p.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    p.add_argument("--size", type=int, default=10, help="Grid size (NxN)")
    p.add_argument("--obst", type=float, default=10.0, help="Percentage of obstacles")
    p.add_argument("--mode", type=str, default="human", choices=["human", "rgb_array"], help="Render mode")
    return p.parse_args()


def main():
    args = parse_args()
    agents = build_agents()
    run_simulation(
        agents=agents,
        size=args.size,
        perc_num_obstacle=args.obst,
        render_mode=args.mode,
        steps=args.steps,
        seed=0,
    )


if __name__ == "__main__":
    main()
