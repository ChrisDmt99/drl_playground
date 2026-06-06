import argparse
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import gymnasium as gym

from scripts.agents.value_agent import ValuePredictionAgent
from utils.utils import read_config_params
from core.policies import policy_evaluation
from utils.plots import plot_decay_schedule, plot_estimation_error, plot_policy_quiver, plot_value_function_heatmap, plot_avg_cumulative_reward
from utils.env_utils import get_grid_shape, get_terminal_states, get_goal_states, get_action_names, get_action_vectors, get_special_states

def run_value_control(config):
    """
    Runs the FrozenLake environment with a Q-Learning agent.
    """
    # FrozenLake environment: initialization
    env = gym.make(config["env_name"], is_slippery=config["is_slippery"], render_mode=config["render_mode"])

    # Compute the list of non-terminal states for later use in random start state selection
    non_terminal_states = []
    num_states = env.observation_space.n
    for s in range(num_states):  
        is_terminal = False
        for action in range(env.action_space.n):
            transitions = env.unwrapped.P[s][action]
            for prob, next_s, reward, terminated in transitions:
                if terminated:
                    is_terminal = True
        
        if not is_terminal:
            non_terminal_states.append(s)

    # Debug
    print(f"Non terminal states: {non_terminal_states}")

    # Agent initialization
    agent = ValuePredictionAgent(
        seed=config["seed"], 
        action_space=env.action_space,
        num_episodes=config["episodes"],
        num_states=env.observation_space.n, 
        num_actions=env.action_space.n, 
        algorithm_params=config["value_agent_params"],
        policy=config["policy"]
    )

    # Compute the optimal value function V of the selected policy
    v_pi = policy_evaluation(pi=agent.policy, P=env.unwrapped.P, gamma=agent.gamma, theta=float(config["value_agent_params"]["theta"]))
    print(f"[Planning Engine] True V* computed successfully.\nGround Truth: {v_pi.round(3)}")

    # Lists to store rewards and estimation errors for plotting
    running_average_rewards = []
    mae_history = []

    # Training loop
    env.reset(seed=config["seed"])
    cumulative_return = 0.0
    pbar = tqdm(range(config["episodes"]), leave=False, desc="Training", unit="episode")
    for ep in pbar:
        pbar.set_postfix(alpha=f"{agent.alphas[ep]:.4f}")

        # We noticed a problem where the MAE between v_pi and v_table doesn't reset as expected.
        # By analyzing the heatmaps of the two v-functions, we noticed that, correctly, by assigning a 
        # fixed and deterministic policy # and always starting from the same state, the agent always follows the same path in the environment.
        # This means that the (predicted) v-function of the states visited by the agent will be estimated correctly, while for the 
        # unvisited states it remains constant. To overcome this problem, the agent must start from a random state at each 
        # episode, thus allowing it to visit all the states.
        start_state = np.random.choice(non_terminal_states)

        # Reset the environment to the initial state
        env.reset()
        env.unwrapped.s = start_state
        state = start_state
        done = False

        # Initialize episode return
        episode_return = 0.0
        discount = 1.0

        # Reset eligibility traces at the beginning of each episode (only for TD-lambda)
        if agent.algorithm_name == "td_lambda":
            agent.reset_traces() 

        while not done:
            # Select an action using the agent's policy
            action, reason = agent.select_action(state)

            # Take the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Compute the TD update for the value function based on the observed transition (state, action, reward, next_state)
            agent.update_value_function(episode=ep, reward=reward, state=state, next_state=next_state, done=done)
            
            # Update episode return
            episode_return += discount * reward
            discount *= agent.gamma

            # Move to the next state
            state = next_state

            # Debug
            if reward > 0:
                print(f"Goal reached: State: {state} -> Chosen Action: {action} | Reward: {reward} | Reason: {reason}")            
        
        # Append rewards and calculate running average reward.
        cumulative_return += episode_return
        running_average_rewards.append(cumulative_return  / (ep + 1))
        
        # Calculate Mean Absolute Error (MAE) between current V-table and V^pi
        current_mae = np.mean(np.abs(v_pi - agent.v_table))
        mae_history.append(current_mae)

        # Let's update the agent's parameters at the end of each episode (e.g., decay epsilon for epsilon-greedy)
        agent.end_of_episode(episode=ep)

    # Debug
    print("Training completed!")

    # Get environment info
    rows, cols = get_grid_shape(env=env)
    terminal_states = get_terminal_states(env=env)
    goal_states = get_goal_states(env=env)
    special_states = get_special_states(env=env)
    action_names = get_action_names(env=env)
    action_vectors = get_action_vectors(env=env)

    # Plotting results
    theoretical_return = np.mean(v_pi[non_terminal_states])
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plot_estimation_error(axs[0, 0], mae_history, table_name="Predicted V vs V^pi")
    plot_avg_cumulative_reward(axs[0, 1], running_average_rewards, title="Average Discounted Return", env=env, theoretical_return=theoretical_return, asymptote_label="Expected Value of Returns")
    plot_decay_schedule(axs[0, 2], agent.alphas, parameter_name="Alpha")

    plot_value_function_heatmap(
        V=v_pi, 
        rows=rows, 
        cols=cols, 
        terminal_states=terminal_states, 
        goal_states=goal_states, 
        special_states=special_states,
        ax=axs[1, 0]
    )    
    axs[1, 0].set_title("Policy Iteration Value Function", fontsize=12, fontweight='bold')
    
    plot_value_function_heatmap(
        V=agent.v_table, 
        rows=rows, 
        cols=cols, 
        terminal_states=terminal_states, 
        goal_states=goal_states, 
        special_states=special_states,    
        ax=axs[1, 1]
    )    
    axs[1, 1].set_title("Predicted Value Function V^pi", fontsize=12, fontweight='bold') 
    
    plot_policy_quiver(
        pi=agent.policy, 
        rows=rows, 
        cols=cols, 
        terminal_states=terminal_states, 
        goal_states=goal_states, 
        special_states=special_states, 
        action_vectors=action_vectors, 
        ax=axs[1, 2]
    )    
    axs[1, 2].set_title("Evaluated Target Policy Trajectories", fontsize=12, fontweight='bold')
    
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