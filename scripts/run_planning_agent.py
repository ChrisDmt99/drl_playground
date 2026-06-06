import argparse

from matplotlib import pyplot as plt
import gymnasium as gym

from scripts.agents.planning_agent import PlanningAgent
from utils.utils import read_config_params
from utils.plots import plot_value_function_heatmap, plot_q_function_heatmap, plot_policy_quiver
from utils.env_utils import get_grid_shape, get_terminal_states, get_goal_states, get_action_names, get_action_vectors, get_special_states

def run_planning_agent(config):
    """
    Runs the Planning Agent with the specified parameters.

    Args:
        config (dict): A dictionary containing the configuration parameters for the environment and agent.
    """
    # FrozenLake environment: initialization
    env = gym.make(config["env_name"], is_slippery=config["is_slippery"], render_mode=config["render_mode"])

    # Planning Agent initialization
    agent = PlanningAgent(env=env, gamma=config["gamma"], theta=float(config["theta"]))

    # Run the planning agent to compute the optimal V*, Q*, and pi* functions
    V_star, Q_star, pi_star = agent.run_all()

    # Get grid environment info
    rows, cols = get_grid_shape(env=env)
    terminal_states = get_terminal_states(env=env)
    goal_states = get_goal_states(env=env)
    special_states = get_special_states(env=env)
    action_names = get_action_names(env=env)
    action_vectors = get_action_vectors(env=env)

    # Plotting results
    fig = plt.figure(figsize=(16, 11))
    ax_v_function = plt.subplot2grid((2, 2), (0, 0))
    ax_q_function = plt.subplot2grid((2, 2), (0, 1))
    ax_optimal_policy = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    plot_value_function_heatmap(
        V=V_star, 
        rows=rows, 
        cols=cols, 
        terminal_states=terminal_states, 
        goal_states=goal_states, 
        special_states=special_states, 
        ax=ax_v_function
    )
    plot_q_function_heatmap(
        Q=Q_star, 
        ax=ax_q_function, 
        action_names=action_names,
        terminal_states=terminal_states, 
        goal_states=goal_states, 
        special_states=special_states
    )
    plot_policy_quiver(
        pi=pi_star, 
        rows=rows, 
        cols=cols, 
        terminal_states=terminal_states, 
        goal_states=goal_states, 
        special_states=special_states, 
        action_vectors=action_vectors, 
        ax=ax_optimal_policy
    )
    plt.tight_layout()
    plt.show()

    # Close the environment
    env.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Planning Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/planning_agent.yaml",
        help="Path to the configuration file (default: configs/planning_agent.yaml)"
    )
    args = parser.parse_args()

    # Read parameters from the YAML configuration file
    planning_params = read_config_params(file_path=args.config)

    # Run the Planning Agent with the specified parameters
    run_planning_agent(config=planning_params)


