import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scripts.environments.bandit_env import MultiArmedBandit
from scripts.agents.policy_agent import PolicyAgent
from utils.utils import read_config_params
from utils.plots import plot_avg_cumulative_reward, plot_decay_schedule, plot_estimation_error, plot_total_regret

def run_bandit(config):
    """
    Runs the Multi-Armed Bandit environment with a policy-based agent.

    Args:
        config (dict): A dictionary containing the configuration parameters for the environment and agent.
    """
    # Bandit environment initialization
    env = MultiArmedBandit(
        num_arms=config["num_arms"],
        seed=config["seed"],
        render_mode=config["render_mode"]
    )

    # Policy Agent initialization   
    agent = PolicyAgent(
        seed=config["seed"], 
        num_states=env.observation_space.n, 
        num_actions=env.action_space.n, 
        num_episodes=config["episodes"],
        policy_name=config["policy_agent_params"]["policy"], 
        policy_params=config['policy_agent_params'][config['policy_agent_params']['policy'] + '_params']
    )

    # Lists to track metrics for plotting over time
    rewards_history = []
    running_average_rewards = []
    mae_history = []
    regret_history = []       
    total_regret_history = []

    # Compute the maximum possible reward for regret calculation (the best arm's expected reward)
    max_possible_reward = float(np.max(env.true_probabilities))

    # Training loop
    pbar = tqdm(range(config["episodes"]), leave=True, desc="Training", unit="episode")
    for ep in pbar:
        if config["policy_agent_params"]["policy"] == "epsilon_greedy":
            pbar.set_postfix(epsilon=f"{agent.epsilons[ep]:.2f}")
            
        elif config["policy_agent_params"]["policy"] == "softmax":
            pbar.set_postfix(temperature=f"{agent.temperatures[ep]:.2f}")

        state, info = env.reset()
        done = False 
        episode_reward = 0
        episode_regret = 0

        while not done:
            # Select an action using the agent's policy
            action, reason = agent.select_action(ep, state, env.action_space)

            # Take the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update regret: the difference between the optimal action's expected reward 
            # and the chosen action's expected reward
            action_expected_reward = env.true_probabilities[action]
            episode_regret += (max_possible_reward - action_expected_reward)

            # Update the agent's Q-table based on the observed reward
            agent.update(state, action, reward)

            # Update episode reward
            episode_reward += reward

            # Update state
            state = next_state

        # Append rewards and calculate running average reward.
        # The cumulative moving average of the episode is calculated by 
        # dividing the total sum of historical awards by the number of current episodes (ep + 1)
        rewards_history.append(episode_reward)
        running_average_rewards.append(np.sum(rewards_history) / (ep + 1))
        
        # Append regret for the current episode and total regret up to the current episode
        regret_history.append(episode_regret)
        total_regret_history.append(np.sum(regret_history))

        # Calculate Mean Absolute Error (MAE) between current Q-table and true probabilities
        current_mae = np.mean(np.abs(env.true_probabilities - agent.q_table[:env.observation_space.n]))
        mae_history.append(current_mae)

    # Debug
    print("Training completed!")

    # Plotting results
    if agent.policy_name in ["epsilon_greedy", "softmax"]:
        fig = plt.figure(figsize=(14, 10))
        ax_error = plt.subplot2grid((2, 2), (0, 0))
        ax_decay = plt.subplot2grid((2, 2), (0, 1))
        ax_reward = plt.subplot2grid((2, 2), (1, 0))
        ax_regret = plt.subplot2grid((2, 2), (1, 1)) 
        
        plot_estimation_error(ax_error, mae_history, table_name="Q-Table")
        
        if agent.policy_name == "epsilon_greedy":
            plot_decay_schedule(ax_decay, agent.epsilons, parameter_name="Epsilon")
        else:
            plot_decay_schedule(ax_decay, agent.temperatures, parameter_name="Temperature")
            
        plot_avg_cumulative_reward(ax_reward, running_average_rewards, title="Average Cumulative Reward", env=env, theoretical_return=max_possible_reward, asymptote_label="Optimal Expected Reward")
        plot_total_regret(ax_regret, total_regret_history)
        
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(14, 8))
        ax_error = plt.subplot2grid((2, 2), (0, 0))
        ax_regret = plt.subplot2grid((2, 2), (0, 1))
        ax_reward = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        plot_estimation_error(ax_error, mae_history, table_name="Q-Table")
        plot_total_regret(ax_regret, total_regret_history)
        plot_avg_cumulative_reward(ax_reward, running_average_rewards, title="Average Cumulative Reward", env=env, theoretical_return=max_possible_reward, asymptote_label="Optimal Expected Reward")
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
    bandit_params = read_config_params(file_path=args.config)

    # Run the Multi-Armed Bandit environment with the specified parameters
    run_bandit(config=bandit_params)


