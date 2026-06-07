import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import core.policies as policies 
from tqdm import tqdm
import gymnasium as gym
from scripts.utils.schedulers import linear_decay_schedule, exponential_decay_schedule, logarithmic_decay_schedule
from scripts.utils.utils import generate_trajectory

class ControlAgent:
    """
    A policy-based agent that selects actions based on a specified policy 
    and updates its Q-table using Reinforcement Learning control algorithms.
    """
    def __init__(
            self, 
            seed: int, 
            num_states: int, 
            num_actions: int, 
            config: Dict[str, Any], 
        ) -> None:
        """
        Initialize the ControlAgent with the given parameters and configurations.

        Args:
            seed (int): The random seed for reproducibility.
            num_states (int): The total number of states in the environment.
            num_actions (int): The total number of actions available in the environment.
            config (Dict[str, Any]): A dictionary containing parameters for algorithms and schedules.
        """
        # Set the random seed for reproducibility
        np.random.seed(seed)
        self.np_random = np.random.default_rng(seed)

        # Set the agent's parameters
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_episodes = config["episodes"]
        self.episodes = range(self.num_episodes)
        self.policy_name = config["policy"]
        self.policy_params = config[config["policy"] + "_params"]  
        self.init_alpha = config["init_alpha"]
        self.min_alpha = config["min_alpha"]
        self.alpha_decay_rate = config["alpha_decay_rate"]
        self.alpha_decay_law = config["alpha_decay_law"] 
        self.gamma = config["gamma"]
        self.max_steps = config["max_steps"]
        self.control_algorithm = config["control_algorithm"]
        
        if self.control_algorithm == "mc_control":
            self.first_visit_mc = config[config["control_algorithm"] + "_params"]["first_visit"]

        # Counters for UCB and Thompson Sampling policies
        self.state_count = np.zeros(self.num_states, dtype=np.int32)
        self.action_counts = np.zeros((self.num_states, self.num_actions), dtype=np.int32)

        # Pre-computation of learning rates for each episode    
        self.alphas = self.init_scheduler(
            init_val=self.init_alpha, 
            min_val=self.min_alpha, 
            rate=self.alpha_decay_rate, 
            law=self.alpha_decay_law
        )  

        # Set policy-specific parameters
        # Epsilon Greedy Policy
        if self.policy_name == "epsilon_greedy":
            self.init_epsilon = self.policy_params["init_epsilon"]
            self.min_epsilon = self.policy_params["min_epsilon"]
            self.epsilon_decay_rate = self.policy_params["decay_rate"]
            self.epsilon_decay_law = self.policy_params["decay_law"]
            
            self.epsilons = self.init_scheduler(
                init_val=self.init_epsilon, 
                min_val=self.min_epsilon, 
                rate=self.epsilon_decay_rate, 
                law=self.epsilon_decay_law
            )
        # Softmax Policy
        elif self.policy_name == "softmax":
            self.init_temperature = self.policy_params["init_temperature"]
            self.min_temperature = self.policy_params["min_temperature"]
            self.temperature_decay_rate = self.policy_params["decay_rate"]
            self.temperature_decay_law = self.policy_params["decay_law"]
            
            self.temperatures = self.init_scheduler(
                init_val=self.init_temperature, 
                min_val=self.min_temperature,  
                rate=self.temperature_decay_rate, 
                law=self.temperature_decay_law
            )
        # Upper Bound Confidence Policy
        elif self.policy_name == "ucb":
            self.c = self.policy_params["c"]
        # Thompson Sampling Policy
        elif self.policy_name == "thompson_sampling":
            self.ts_alpha = self.policy_params["alpha"]
            self.ts_beta = self.policy_params["beta"]
        else:
            raise ValueError(f"Invalid policy name: {self.policy_name}")

        # Initialize Q-table, V-function and final policy placeholders
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)
        self.v_function = np.zeros(num_states, dtype=np.float32)
        self.final_policy = np.zeros(num_states, dtype=np.int32)

        # History structures defined as instance attributes
        self.mae_history = [0.0] * self.num_episodes
        self.rewards_history = [0.0] * self.num_episodes
        self.running_average_rewards = [0.0] * self.num_episodes
        self.total_regret_history = [0.0] * self.num_episodes

    def init_scheduler(self, init_val: float, min_val: float, rate: float, law: str) -> np.ndarray:
        """
        Initializes a decay schedule array for learning rates or exploration parameters.

        Args:
            init_val (float): The starting value of the parameter.
            min_val (float): The minimum floor value for the parameter.
            rate (float): The decay rate applied per episode.
            law (str): The decay law to use ('linear', 'exponential', 'logarithmic').

        Returns:
            np.ndarray: An array containing pre-computed values for each episode.
        """
        if law == 'linear':
            return linear_decay_schedule(init_value=init_val, min_value=min_val, num_episodes=self.num_episodes)
        elif law == 'exponential':
            return exponential_decay_schedule(init_value=init_val, min_value=min_val, decay_rate=rate, num_episodes=self.num_episodes)
        elif law == 'logarithmic':
            return logarithmic_decay_schedule(init_value=init_val, min_value=min_val, decay_rate=rate, num_episodes=self.num_episodes)
        else:
            raise ValueError(f"Invalid decay law: {law}")

    def select_action(self, episode: int, q_values: np.ndarray, state: int, action_space: gym.Space) -> Tuple[int, str]:
        """
        Selects an action based on the current policy selection method and the Q-table.

        Args:
            episode (int): The current training episode index.
            state (int): The current state of the environment.
            action_space (gym.Space): The action space of the environment.

        Returns:
            Tuple[int, str]: A tuple containing the selected action index and the reason string.
        """  
        if self.policy_name == "epsilon_greedy":
            return policies.epsilon_greedy_policy(q_values, self.epsilons[episode], action_space, np_random=self.np_random)
        
        elif self.policy_name == "softmax":
            return policies.softmax_policy(q_values, self.temperatures[episode], np_random=self.np_random)
        
        elif self.policy_name == "ucb":
            # Avoid division by zero if the state has never been visited before
            state_total = max(1, self.state_count[state])
            return policies.ucb_policy(q_values, self.c, state_total, self.action_counts[state], np_random=self.np_random)
        
        elif self.policy_name == "thompson_sampling":
            return policies.thompson_sampling_policy(q_values, self.ts_alpha, self.ts_beta, self.action_counts[state], np_random=self.np_random)
        
        else:
            raise ValueError("Unsupported policy")
    
    def run(self, env: gym.Env, Q_star: Optional[np.ndarray], V_star: Optional[np.ndarray]) -> None:
        """
        Executes the training loop using the specified control algorithm.

        Args:
            env (gym.Env): The Gymnasium environment instance.
            Q_star (Optional[np.ndarray]): The optimal Q-table benchmark for MAE calculation.
            V_star (Optional[np.ndarray]): The optimal Value function benchmark for Regret calculation.
        """
        if self.control_algorithm == "mc_control": 
            self.run_mc_agent(env, Q_star, V_star)
        elif self.control_algorithm == "sarsa":
            self.run_sarsa_agent(env, Q_star, V_star)
        elif self.control_algorithm == "q_learning":
            self.run_q_learning_agent(env, Q_star, V_star)
        elif self.control_algorithm == "double_q_learning":
            self.run_double_q_learning_agent(env, Q_star, V_star)
        else:
            raise ValueError("Unsupported control algorithm")

    def run_mc_agent(self, env: gym.Env, Q_star: Optional[np.ndarray], V_star: Optional[np.ndarray]) -> None:
        """
        Trains the agent using the First/Every Visit Monte Carlo Control algorithm.
        """
        # Pre-computation of discounts for the return values
        self.discounts = np.logspace(0, self.max_steps, num=self.max_steps, base=self.gamma, endpoint=False)

        # Training loop parameters
        cumulative_regret = 0.0
        total_rewards_earned = 0.0
        visited = np.zeros((self.num_states, self.num_actions), dtype=bool)
        pbar = tqdm(self.episodes, leave=True, desc="MC Control Training", unit="episode")        
        for ep in pbar:
            if self.policy_name == "epsilon_greedy":
                pbar.set_postfix(epsilon=f"{self.epsilons[ep]:.2f}")
            elif self.policy_name == "softmax":
                pbar.set_postfix(temperature=f"{self.temperatures[ep]:.2f}")
            
            # Generate trajectory
            trajectory = generate_trajectory(
                agent_select_action_fn=lambda s: self.select_action(episode=ep, q_values=self.q_table[s], state=s, action_space=env.action_space)[0], 
                env=env, 
                max_steps=self.max_steps
            )
            traj_len = len(trajectory)
            if traj_len == 0:
                continue
            
            # Rewards
            all_rewards = np.array([step[2] for step in trajectory], dtype=np.float32)

            # Calculating historical metrics
            episode_reward = float(np.sum(all_rewards))
            self.rewards_history[ep] = episode_reward
            total_rewards_earned += episode_reward
            self.running_average_rewards[ep] = total_rewards_earned / (ep + 1)

            # Cumulative Regret calculation based on the starting state of the current episode
            if V_star is not None:
                initial_state = trajectory[0][0]
                discounted_episode_reward = np.sum(self.discounts[:traj_len] * all_rewards)
                episode_regret = V_star[initial_state] - discounted_episode_reward
                # cumulative_regret += max(0.0, episode_regret)
                cumulative_regret += episode_regret
            self.total_regret_history[ep] = cumulative_regret

            # Reset the visited actions-states tracker for First-Visit setup
            if self.first_visit_mc:
                visited.fill(False)
            
            # For each timestamp of the trajectory
            for t, (state, action, _, _, _) in enumerate(trajectory):
                # MC Control: First Visit Logic
                if self.first_visit_mc:
                    if visited[state][action]:
                        continue
                    visited[state][action] = True

                # Updated exploration counters for UCB and Thompson Sampling 
                self.state_count[state] += 1
                self.action_counts[state, action] += 1

                # Remaining steps to the end of the trajectory
                n_steps = traj_len - t

                # Computing the discounted cumulative return
                _return = np.sum(self.discounts[:n_steps] * all_rewards[t:])

                # Update of the Q-table
                self.q_table[state, action] += self.alphas[ep] * (_return - self.q_table[state, action])
            
            # Computing the MAE on the Q-table
            if Q_star is not None:
                self.mae_history[ep] = float(np.mean(np.abs(Q_star - self.q_table)))
            else:
                self.mae_history[ep] = float(0.0)

        # Compute the V-function and the policy corresponding to the final Q-table
        self.v_function = np.max(self.q_table, axis=1)
        self.final_policy = np.argmax(self.q_table, axis=1)

    def run_sarsa_agent(self, env: gym.Env, Q_star: Optional[np.ndarray], V_star: Optional[np.ndarray]) -> None:
        """
        Trains the agent using the On-Policy SARSA (State-Action-Reward-State-Action) algorithm.
        """
        # Training loop parameters
        cumulative_regret = 0.0
        total_rewards_earned = 0.0
        pbar = tqdm(self.episodes, leave=True, desc="SARSA Training", unit="episode")
        
        for ep in pbar:
            if self.policy_name == "epsilon_greedy":
                pbar.set_postfix(epsilon=f"{self.epsilons[ep]:.2f}")
            elif self.policy_name == "softmax":
                pbar.set_postfix(temperature=f"{self.temperatures[ep]:.2f}")

            state, _ = env.reset()
            initial_state = state
            done = False

            episode_reward = 0.0
            discounted_episode_reward = 0.0
            current_gamma = 1.0

            action, _ = self.select_action(episode=ep, q_values=self.q_table[state], state=state, action_space=env.action_space)

            # Iterate until the end of the episode
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_action, _ = self.select_action(episode=ep, q_values=self.q_table[next_state], state=next_state, action_space=env.action_space)

                # Reward and discounted return tracking
                episode_reward += reward
                discounted_episode_reward += current_gamma * reward
                current_gamma *= self.gamma

                # Updated exploration counters for UCB and Thompson Sampling 
                self.state_count[state] += 1
                self.action_counts[state, action] += 1

                # SARSA update of the Q-table
                td_target = reward + self.gamma * self.q_table[next_state, next_action] * (not terminated)
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alphas[ep] * td_error

                # Update the action and the state for the next iteration
                state, action = next_state, next_action

            # Compute of the total regret based on initial state and discounted return
            if V_star is not None:
                episode_regret = V_star[initial_state] - discounted_episode_reward
                # cumulative_regret += max(0.0, episode_regret)
                cumulative_regret += episode_regret
            self.total_regret_history[ep] = cumulative_regret

            # Computing the running average reward via continuous tracking optimization
            self.rewards_history[ep] = episode_reward
            total_rewards_earned += episode_reward
            self.running_average_rewards[ep] = total_rewards_earned / (ep + 1)

            # Computing the MAE on the Q-table
            if Q_star is not None:
                self.mae_history[ep] = float(np.mean(np.abs(Q_star - self.q_table)))
            else:
                self.mae_history[ep] = float(0.0)

        # Compute the V-function and the policy corresponding to the final Q-table
        self.v_function = np.max(self.q_table, axis=1)
        self.final_policy = np.argmax(self.q_table, axis=1)

    def run_q_learning_agent(self, env: gym.Env, Q_star: Optional[np.ndarray], V_star: Optional[np.ndarray]) -> None:
        """
        Trains the agent using the Off-Policy Q-Learning algorithm.
        """
        # Training loop parameters
        cumulative_regret = 0.0
        total_rewards_earned = 0.0
        pbar = tqdm(self.episodes, leave=True, desc="Q-Learning Training", unit="episode")
        for ep in pbar:
            if self.policy_name == "epsilon_greedy":
                pbar.set_postfix(epsilon=f"{self.epsilons[ep]:.2f}")
            elif self.policy_name == "softmax":
                pbar.set_postfix(temperature=f"{self.temperatures[ep]:.2f}")

            state, _ = env.reset()
            initial_state = state
            done = False

            episode_reward = 0.0
            discounted_episode_reward = 0.0
            current_gamma = 1.0

            # Iterate until the end of the episode
            while not done:       
                # Select action for the current state (Off-policy flavor)         
                action, _ = self.select_action(episode=ep, q_values=self.q_table[state], state=state, action_space=env.action_space)

                # Environment step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Reward and discounted return tracking
                episode_reward += reward
                discounted_episode_reward += current_gamma * reward
                current_gamma *= self.gamma

                # Updated exploration counters for UCB and Thompson Sampling 
                self.state_count[state] += 1
                self.action_counts[state, action] += 1
                
                # Q-learning update of the Q-table
                td_target = reward + self.gamma * self.q_table[next_state].max() * (not terminated)
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alphas[ep] * td_error

                # Update the state for the next iteration
                state = next_state

            # Compute of the total regret based on initial state and discounted return
            if V_star is not None:
                episode_regret = V_star[initial_state] - discounted_episode_reward
                # cumulative_regret += max(0.0, episode_regret)
                cumulative_regret += episode_regret
            self.total_regret_history[ep] = cumulative_regret

            # Computing the running average reward via continuous tracking optimization
            self.rewards_history[ep] = episode_reward
            total_rewards_earned += episode_reward
            self.running_average_rewards[ep] = total_rewards_earned / (ep + 1)

            # Computing the MAE on the Q-table
            if Q_star is not None:
                self.mae_history[ep] = float(np.mean(np.abs(Q_star - self.q_table)))
            else:
                self.mae_history[ep] = float(0.0)

        # Compute the V-function and the policy corresponding to the final Q-table
        self.v_function = np.max(self.q_table, axis=1)
        self.final_policy = np.argmax(self.q_table, axis=1)

    def run_double_q_learning_agent(self, env: gym.Env, Q_star: Optional[np.ndarray], V_star: Optional[np.ndarray]) -> None:
        """
        Trains the agent using the Off-Policy Double Q-Learning algorithm.
        """
        # Initialize the Q-tables Q1 and Q2
        Q1 = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        Q2 = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        # Training loop parameters
        cumulative_regret = 0.0
        total_rewards_earned = 0.0
        pbar = tqdm(self.episodes, leave=True, desc="Double Q-Learning Training", unit="episode")
        for ep in pbar:
            if self.policy_name == "epsilon_greedy":
                pbar.set_postfix(epsilon=f"{self.epsilons[ep]:.2f}")
            elif self.policy_name == "softmax":
                pbar.set_postfix(temperature=f"{self.temperatures[ep]:.2f}")

            state, _ = env.reset()
            initial_state = state
            done = False

            episode_reward = 0.0
            discounted_episode_reward = 0.0
            current_gamma = 1.0

            # Iterate until the end of the episode
            while not done:
                # Select action for the current state (Off-policy flavor)         
                action, _ = self.select_action(episode=ep, q_values=(Q1[state]+Q2[state])*0.5, state=state, action_space=env.action_space)

                # Environment step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # DEBUG
                if terminated:
                    print("Terminated")
                if truncated:
                    print("Truncated")

                # Reward and discounted return tracking
                episode_reward += reward
                discounted_episode_reward += current_gamma * reward
                current_gamma *= self.gamma

                # Updated exploration counters for UCB and Thompson Sampling 
                self.state_count[state] += 1
                self.action_counts[state, action] += 1

                # Q-learning update of the Q-table
                if np.random.rand() < 0.5:
                    argmax_Q1 = np.argmax(Q1[next_state])
                    td_target = reward + self.gamma * Q2[next_state, argmax_Q1] * (not terminated)
                    td_error = td_target - Q1[state, action]
                    Q1[state, action] += self.alphas[ep] * td_error

                else:
                    argmax_Q2 = np.argmax(Q2[next_state])
                    td_target = reward + self.gamma * Q1[next_state, argmax_Q2] * (not terminated)
                    td_error = td_target - Q2[state, action]
                    Q2[state, action] += self.alphas[ep] * td_error

                # Update the state for the next iteration
                state = next_state

            # Update the Q-table as the mean of Q1 and Q2
            self.q_table = (Q1 + Q2) / 2.0

            # Compute of the total regret based on initial state and discounted return
            if V_star is not None:
                episode_regret = V_star[initial_state] - discounted_episode_reward
                # cumulative_regret += max(0.0, episode_regret)
                cumulative_regret += episode_regret
            self.total_regret_history[ep] = cumulative_regret

            # Computing the running average reward via continuous tracking optimization
            self.rewards_history[ep] = episode_reward
            total_rewards_earned += episode_reward
            self.running_average_rewards[ep] = total_rewards_earned / (ep + 1)

            # Computing the MAE on the Q-table
            if Q_star is not None:
                self.mae_history[ep] = float(np.mean(np.abs(Q_star - self.q_table)))
            else:
                self.mae_history[ep] = float(0.0)

        # Compute the V-function and the policy corresponding to the final Q-table
        self.v_function = np.max(self.q_table, axis=1)
        self.final_policy = np.argmax(self.q_table, axis=1)