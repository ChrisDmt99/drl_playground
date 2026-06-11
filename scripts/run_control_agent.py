import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym

from scripts.agents.control_agent import ControlAgent
from utils.utils import read_config_params
from utils.plots import plot_avg_cumulative_reward, plot_decay_schedule, plot_estimation_error, plot_total_regret
from core.value_functions import compute_optimal_v_function
from core.policies import compute_optimal_policy
from core.q_functions import compute_optimal_q_function
from utils.plots import plot_value_function_heatmap, plot_q_function_heatmap, plot_policy_quiver
from utils.env_utils import get_grid_shape, get_terminal_states, get_goal_states, get_action_names, get_action_vectors, get_special_states
import matplotlib.patches as mpatches

def run_control_agent(config):
    """
    Runs the Taxi environment with a control agent.

    Args:
        config (dict): A dictionary containing the configuration parameters for the environment and agent.
    """
    # Taxi environment initialization
    env = gym.make(
        config["env_name"], 
        is_rainy=config["is_rainy"], 
        rainy_probability=config["rainy_probability"], 
        fickle_passenger=config["fickle_passenger"], 
        fickle_probability=config["fickle_probability"], 
        render_mode=config["render_mode"]
    )

    # Identify valid and unreachable states 
    # Generate a binary mask to isolate states where passenger starting point equals destination
    valid_states_mask = np.ones(env.observation_space.n, dtype=bool)
    for s in range(env.observation_space.n):
        taxi_row, taxi_col, passenger_look, destination = env.unwrapped.decode(s)
        if passenger_look == destination:
            valid_states_mask[s] = False
            
    valid_state_indices = np.where(valid_states_mask)[0]
    invalid_state_indices = np.where(~valid_states_mask)[0]

    # Computing the optimal Q-table and the corresponding optimal V-function
    Q_star = compute_optimal_q_function(env, gamma=config["gamma"], theta=float(config["theta"]))
    V_star = compute_optimal_v_function(env, gamma=config["gamma"], theta=float(config["theta"]))
    pi_star = compute_optimal_policy(env, gamma=config["gamma"], theta=float(config["theta"]))

    # Policy Agent initialization   
    agent = ControlAgent(
        seed=config["seed"], 
        num_states=env.observation_space.n, 
        num_actions=env.action_space.n, 
        config=config,
        valid_states=valid_state_indices
    )

    # Compute the maximum possible reward for regret calculation
    max_possible_reward = float(V_star[0])

    # Training loop
    agent.run(env=env, Q_star=Q_star, V_star=V_star)

    # Debug
    print("Training completed!")

    # Set seaborn theme for plot style
    sns.set_theme(style="whitegrid")

    # Figure 1: Performance metrics and decay schedule plots
    if agent.policy_name in ["epsilon_greedy", "softmax"]:
        fig1 = plt.figure(figsize=(15, 11))
        ax_error = plt.subplot2grid((2, 2), (0, 0))
        ax_decay = plt.subplot2grid((2, 2), (0, 1))
        ax_reward = plt.subplot2grid((2, 2), (1, 0))
        ax_regret = plt.subplot2grid((2, 2), (1, 1)) 
        
        plot_estimation_error(ax_error, agent.mae_history, table_name="Q-Table")
        
        # Plot parameter decay schedules (Epsilon/Temperature)
        if agent.policy_name == "epsilon_greedy":
            plot_decay_schedule(ax_decay, agent.epsilons, parameter_name="Epsilon")
        else:
            plot_decay_schedule(ax_decay, agent.temperatures, parameter_name="Temperature")
        
        # Overlay the learning rate (Alpha) decay schedule onto the same plot
        ax_decay.plot(range(len(agent.alphas)), agent.alphas, color='purple', linestyle='--', label="Alpha (Learning Rate)")
        ax_decay.legend()
            
        plot_avg_cumulative_reward(ax_reward, agent.running_average_rewards, title="Average Cumulative Reward", env=env, theoretical_return=max_possible_reward, asymptote_label="Optimal Expected Reward")
        plot_total_regret(ax_regret, agent.total_regret_history)
        
        plt.suptitle(f"Training Analysis (Valid States Only) - Policy: {agent.policy_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        fig1 = plt.figure(figsize=(15, 9))
        ax_error = plt.subplot2grid((2, 2), (0, 0))
        ax_regret = plt.subplot2grid((2, 2), (0, 1))
        ax_reward = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        
        plot_estimation_error(ax_error, agent.mae_history, table_name="Q-Table")
        plot_total_regret(ax_regret, agent.total_regret_history)
        plot_avg_cumulative_reward(ax_reward, agent.running_average_rewards, title="Average Cumulative Reward", env=env, theoretical_return=max_possible_reward, asymptote_label="Optimal Expected Reward")
        
        plt.suptitle(f"Training Analysis (Valid States Only) - Policy: {agent.policy_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # --- Impossibible states mask generation for heatmaps ---
    # Compute error representations for both Q-table and V-function
    q_error = (agent.q_table - Q_star) ** 2                    
    v_error = ((agent.v_function - V_star) ** 2).reshape(-1, 1) 

    # Create boolean masks tracking unreachable/impossible environment states
    invalid_mask_q = np.zeros_like(q_error, dtype=bool)
    invalid_mask_v = np.zeros_like(v_error, dtype=bool)

    for s in range(env.observation_space.n):
        if not valid_states_mask[s]:
            invalid_mask_q[s, :] = True  # Mask entire state row for Q-table
            invalid_mask_v[s, 0] = True  # Mask state row for V-function

    # Figure 2 & 3 Combined: Heatmaps of Q-Table and V-Function Estimation Errors
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.5]) 
    
    # Connect the axes using the newly created grid 
    ax_q = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])

    # Define the state configuration for the Taxi environment
    unreachable_states = invalid_state_indices.tolist()
    action_names = ["South (0)", "North (1)", "East (2)", "West (3)", "Pick (4)", "Drop (5)"]
    
    # States where the passenger is visually located at the destination (Goal states)
    goal_states = [s for s in range(env.observation_space.n) if env.unwrapped.decode(s)[2] == env.unwrapped.decode(s)[3]]

    # Plot the error associated with the V-function
    plot_value_function_heatmap(
        V=v_error, 
        ax=ax_v,
        rows=env.observation_space.n, 
        cols=1, 
        terminal_states=[], 
        goal_states=goal_states, 
        special_states=unreachable_states, 
        show_text=False
    )
    # Update titles and axes to reflect estimation error rather than standard V-function values
    ax_v.set_title("V-Function Squared Error vs V*", fontweight="bold")

    # Plot the error associated with the Q-table
    plot_q_function_heatmap(
        Q=q_error, 
        ax=ax_q, 
        action_names=action_names,
        terminal_states=[], 
        goal_states=goal_states, 
        special_states=unreachable_states,
        show_text=False
    )
    # Update the title to reflect the Q-table estimation error
    ax_q.set_title("Q-Table Squared Error vs Optimal Q*", fontweight="bold")

    # Manually add legend patches for explanatory clarity
    unreachable_patch = mpatches.Patch(color='#8d6e63', label='[S] Unreachable States (Excluded)')
    goal_patch = mpatches.Patch(color='#4caf50', label='[G] Goal States')
    ax_q.legend(handles=[unreachable_patch, goal_patch], loc='upper right')

    plt.suptitle("Functional Estimation Errors Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Close the environment session securely
    env.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Multi-Armed Bandit environment with policy-based agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/policy_agent.yaml",
        help="Path to the configuration file (default: configs/policy_agent.yaml)"
    )
    args = parser.parse_args()

    # Read parameters from the YAML configuration file
    control_params = read_config_params(file_path=args.config)

    # Run the Taxi-v4 environment with the specified parameters
    run_control_agent(config=control_params)