import argparse
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import gymnasium as gym

from scripts.agents.value_agent import ValuePredictionAgent
from utils.utils import read_config_params
from core.value_functions import compute_optimal_v_function
from utils.plots import plot_avg_cumulative_reward, plot_decay_schedule, plot_estimation_error

from core.policies import compute_optimal_policy
from core.q_functions import compute_optimal_q_function

def run_value_control(config):
    """
    Runs the FrozenLake environment with a Q-Learning agent.
    """
    # FrozenLake environment: initialization
    env = gym.make(config["env_name"], is_slippery=config["is_slippery"], render_mode=config["render_mode"])

    # Agent initialization
    agent = ValuePredictionAgent(
        seed=config["seed"], 
        action_space=env.action_space,
        num_episodes=config["episodes"],
        num_states=env.observation_space.n, 
        num_actions=env.action_space.n, 
        algorithm_params=config["value_agent_params"]
    )

    # Compute the true optimal value function V* using the environment's transition probabilities and rewards (planning engine)
    v_star = compute_optimal_v_function(env, gamma=agent.gamma, theta=float(config["value_agent_params"]["theta"]))
    print(f"[Planning Engine] True V* computed successfully.\nGround Truth: {v_star.round(3)}")

    # Debug
    policy_star = compute_optimal_policy(env, gamma=agent.gamma, theta=float(config["value_agent_params"]["theta"]))
    q_star = compute_optimal_q_function(env, gamma=agent.gamma, theta=float(config["value_agent_params"]["theta"]))
    print(f"[Planning Engine] Optimal Policy computed successfully.\nGround Truth: {[policy_star[s] for s in range(env.observation_space.n)]}")
    print(f"[Planning Engine] Optimal Q* computed successfully.\nGround Truth:\n{q_star.round(3)}")

    # Lists to store rewards and estimation errors for plotting
    rewards_history = []
    running_average_rewards = []
    mae_history = []

    # Training loop
    pbar = tqdm(range(config["episodes"]), leave=False, desc="Training", unit="episode")
    for ep in pbar:
        pbar.set_postfix(alpha=f"{agent.alphas[ep]:.4f}")

        # Reset the environment to the initial state
        state, info = env.reset(seed=config["seed"] + ep)
        done = False

        # Initialize episode reward
        episode_reward = 0

        # Reset eligibility traces at the beginning of each episode (only for TD-lambda)
        if agent.algorithm_name == "td_lambda":
            agent.reset_traces() 

        while not done:
            # Select an action using the agent's policy
            action, reason = agent.select_action(state, env.unwrapped.P)

            # Take the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Compute the TD update for the value function based on the observed transition (state, action, reward, next_state)
            agent.update_value_function(episode=ep, reward=reward, state=state, next_state=next_state, done=done)
            
            # Update episode reward
            episode_reward += reward

            # Move to the next state
            state = next_state

            # Debug
            if reward > 0:
                print(f"Goal reached: State: {state} -> Chosen Action: {action} | Reward: {reward} | Reason: {reason}")            
        
        # Append rewards and calculate running average reward.
        rewards_history.append(episode_reward)
        running_average_rewards.append(np.sum(rewards_history) / (ep + 1))
        
        # Calculate Mean Absolute Error (MAE) between current V-table and true V*
        current_mae = np.mean(np.abs(v_star - agent.v_table))
        mae_history.append(current_mae)

        # Let's update the agent's parameters at the end of each episode (e.g., decay epsilon for epsilon-greedy)
        agent.end_of_episode(episode=ep)

    # Debug
    print("Training completed!")

    # Plotting results
    max_possible_reward = env.spec.reward_threshold
    fig = plt.figure(figsize=(12, 10))
    ax_error = plt.subplot2grid((2, 2), (0, 0))
    ax_decay = plt.subplot2grid((2, 2), (0, 1))
    ax_reward = plt.subplot2grid((2, 2), (1, 0), colspan=2)  
    plot_estimation_error(ax_error, mae_history, table_name="V-Table")
    plot_decay_schedule(ax_decay, agent.alphas, parameter_name="Alpha")
    plot_avg_cumulative_reward(ax_reward, running_average_rewards, env, max_possible_reward=max_possible_reward)

    plt.tight_layout()
    plt.show()
    
    # Close the environment
    env.close()    

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run FrozenLake environment with Q-Learning agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/value_agent.yaml",
        help="Path to the configuration file (default: configs/value_agent.yaml)"
    )
    args = parser.parse_args()

    # Read parameters from the YAML configuration file
    value_control_params = read_config_params(file_path=args.config)

    # Run the FrozenLake environment with the specified parameters
    run_value_control(config=value_control_params)


