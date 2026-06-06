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

def run_control_agent(config):
    """
    Runs the Taxi environment with a control agent.

    Args:
        config (dict): A dictionary containing the configuration parameters for the environment and agent.
    """
    # Taxi environment initialization
    # env = gym.make(
    #     config["env_name"], 
    #     is_rainy=config["is_rainy"], 
    #     rainy_probability=config["rainy_probability"], 
    #     fickle_passenger=config["fickle_passenger"], 
    #     fickle_probability=config["fickle_probability"], 
    #     render_mode=config["render_mode"]
    # )
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=config["render_mode"])

    # Computing the optimal Q-table and the corresponding optimal V-function
    V_star = compute_optimal_v_function(env, gamma=config["gamma"], theta=float(config["theta"]))
    Q_star = compute_optimal_q_function(env, gamma=config["gamma"], theta=float(config["theta"]))
    pi_star = compute_optimal_policy(env, gamma=config["gamma"], theta=float(config["theta"]))

    # Policy Agent initialization   
    agent = ControlAgent(
        seed=config["seed"], 
        num_states=env.observation_space.n, 
        num_actions=env.action_space.n, 
        config=config
    )

    # Compute the maximum possible reward for regret calculation
    max_possible_reward = float(np.mean(V_star))

    # Training loop
    agent.run(env=env, Q_star=Q_star, V_star=V_star)

    # Debug
    print("Training completed!")

    # Lists to track metrics for plotting over time
    mae_history = agent.mae_history
    running_average_rewards = agent.running_average_rewards
    total_regret_history = agent.total_regret_history   

    # Set seaborn theme for a clean and professional plot style
    sns.set_theme(style="whitegrid")

    # 1. PERFORMANCE METRICS AND DECAY SCHEDULES PLOTS
    if agent.policy_name in ["epsilon_greedy", "softmax"]:
        fig1 = plt.figure(figsize=(15, 11))
        ax_error = plt.subplot2grid((2, 2), (0, 0))
        ax_decay = plt.subplot2grid((2, 2), (0, 1))
        ax_reward = plt.subplot2grid((2, 2), (1, 0))
        ax_regret = plt.subplot2grid((2, 2), (1, 1)) 
        
        plot_estimation_error(ax_error, mae_history, table_name="Q-Table")
        
        # Plot parameter decay schedules (Epsilon/Temperature)
        if agent.policy_name == "epsilon_greedy":
            plot_decay_schedule(ax_decay, agent.epsilons, parameter_name="Epsilon")
        else:
            plot_decay_schedule(ax_decay, agent.temperatures, parameter_name="Temperature")
        
        # Overlay the learning rate (Alpha) decay schedule onto the same plot
        ax_decay.plot(range(len(agent.alphas)), agent.alphas, color='purple', linestyle='--', label="Alpha (Learning Rate)")
        ax_decay.plot(range(len(agent.discounts)), agent.discounts, color='green', linestyle='-.', label="Gamma (Discounts)")
        ax_decay.legend()
            
        plot_avg_cumulative_reward(ax_reward, running_average_rewards, title="Average Cumulative Reward", env=env, theoretical_return=max_possible_reward, asymptote_label="Optimal Expected Reward")
        plot_total_regret(ax_regret, total_regret_history)
        
        plt.suptitle(f"Training Analysis - Policy: {agent.policy_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        fig1 = plt.figure(figsize=(15, 9))
        ax_error = plt.subplot2grid((2, 2), (0, 0))
        ax_regret = plt.subplot2grid((2, 2), (0, 1))
        ax_reward = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        
        plot_estimation_error(ax_error, mae_history, table_name="Q-Table")
        plot_total_regret(ax_regret, total_regret_history)
        plot_avg_cumulative_reward(ax_reward, running_average_rewards, title="Average Cumulative Reward", env=env, theoretical_return=max_possible_reward, asymptote_label="Optimal Expected Reward")
        
        plt.suptitle(f"Training Analysis - Policy: {agent.policy_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # 2. FINAL SPATIAL HEATMAPS GENERATION (5x5 GRID REDUCTION)
    print("Generating Final Spatial Heatmaps (5x5 Grid reduction)...")
    
    # Initialize 5x5 matrices representing the physical Taxi map coordinates
    v_spatial = np.zeros((5, 5), dtype=np.float32)
    q_spatial = np.zeros((5, 5), dtype=np.float32)
    visits_spatial = np.zeros((5, 5), dtype=np.float32)
    policy_spatial = np.zeros((5, 5), dtype=np.float32)
    state_counts = np.zeros((5, 5), dtype=np.float32)

    # Iterate over all 500 environment states to decode and aggregate them spatially
    for state in range(env.observation_space.n):
        # unwrap and decode extract: (taxi_row, taxi_col, passenger_location, destination)
        taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
        
        v_spatial[taxi_row, taxi_col] += agent.v_function[state]
        q_spatial[taxi_row, taxi_col] += np.mean(agent.q_table[state]) # Average Q-values of actions at this grid position
        visits_spatial[taxi_row, taxi_col] += agent.state_count[state]
        policy_spatial[taxi_row, taxi_col] += agent.final_policy[state]
        state_counts[taxi_row, taxi_col] += 1.0

    # Compute the average value per grid cell to eliminate passenger/destination combination bias
    v_spatial /= state_counts
    q_spatial /= state_counts
    # For the policy mapping, round to the most predominant action chosen in that physical cell
    policy_spatial = np.round(policy_spatial / state_counts).astype(np.int32)

    # Create the figure subplots for the 4 final state heatmaps
    fig_heat, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Heatmap 1: Final V-Function
    sns.heatmap(v_spatial, annot=True, fmt=".2f", cmap="YlGnBu", ax=axs[0, 0], cbar_kws={'label': 'Value'})
    axs[0, 0].set_title("Spatial Heatmap of Final V-Function ($V$)", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Taxi Column (X)")
    axs[0, 0].set_ylabel("Taxi Row (Y)")

    # Heatmap 2: Final Q-Table (Aggregated Mean Value of Actions)
    sns.heatmap(q_spatial, annot=True, fmt=".2f", cmap="magma", ax=axs[0, 1], cbar_kws={'label': 'Q-Value'})
    axs[0, 1].set_title("Spatial Heatmap of Average Q-Table Values", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Taxi Column (X)")
    axs[0, 1].set_ylabel("Taxi Row (Y)")

    # Heatmap 3: State Visit Frequencies (Exploration Pattern)
    sns.heatmap(visits_spatial, annot=True, fmt=".0f", cmap="Oranges", ax=axs[1, 0], cbar_kws={'label': 'Visits'})
    axs[1, 0].set_title("Spatial Heatmap of State Visits (Exploration)", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Taxi Column (X)")
    axs[1, 0].set_ylabel("Taxi Row (Y)")

    # Heatmap 4: Final Policy
    # Standard numerical action mapping for Taxi-v3/v4 text annotations
    sns.heatmap(policy_spatial, annot=True, fmt="d", cmap="coolwarm", cbar=False, ax=axs[1, 1])
    axs[1, 1].set_title("Map of Predominant Policy Actions ($\pi$)\n(0:South, 1:North, 2:East, 3:West, 4:Pickup, 5:Dropoff)", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("Taxi Column (X)")
    axs[1, 1].set_ylabel("Taxi Row (Y)")

    plt.suptitle("Analysis of Final Internal State Aggregated on 5x5 Map", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # Close the environment
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

    # Run the Multi-Armed Bandit environment with the specified parameters
    run_control_agent(config=control_params)