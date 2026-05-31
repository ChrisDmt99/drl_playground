import argparse
import tqdm
from tqdm import tqdm
import gymnasium as gym
from agents.tabular_agents import StaticAgent
# from core.policies import decay_epsilon

from utils.utils import read_frozen_lake_params


def run_frozenlake(config):
    """
    Runs the FrozenLake environment with a Q-Learning agent.
    """
    # FrozenLake environment: initialization
    env = gym.make(config["env_name"], is_slippery=config["is_slippery"], render_mode=config["render_mode"])

    # Agent initialization
    if config["agent"] == "static":    
        agent = StaticAgent(seed=config["seed"], num_states=env.observation_space.n, num_actions=env.action_space.n, policy_name=config["policy"], policy_params=config.get(f"{config['policy']}_params", {}))
    else:
        raise ValueError(f"Agent type '{config['agent']}' not recognized. Please check the configuration.")

    # Training loop
    pbar = tqdm(range(config["episodes"]), leave=False, desc="Training", unit="episode")
    for ep in pbar:
        if config["policy"] == "epsilon_greedy":
            pbar.set_postfix(epsilon=f"{agent.epsilon:.2f}")
            
        elif config["policy"] == "softmax":
            pbar.set_postfix(temperature=f"{agent.temperature:.2f}")

        state, info = env.reset()
        done = False 

        while not done:
            # Select an action using the agent's policy
            action, reason = agent.select_action(state, env.action_space)

            # Take the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state

            # Debug
            if reward > 0:
                print(f"Goal reached: State: {state} -> Chosen Action: {action} | Reward: {reward} | Reason: {reason}")            

        # Let's update the agent's parameters at the end of each episode (e.g., decay epsilon for epsilon-greedy)
        agent.end_of_episode()

    # Close the environment

    print("Training completed!")
    env.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run FrozenLake environment with Q-Learning agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/frozen_lake.yaml",
        help="Path to the configuration file (default: configs/frozen_lake.yaml)"
    )
    args = parser.parse_args()

    # Read parameters from the YAML configuration file
    frozen_lake_params = read_frozen_lake_params(file_path=args.config)

    # Run the FrozenLake environment with the specified parameters
    run_frozenlake(config=frozen_lake_params)


