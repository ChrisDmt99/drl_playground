import numpy as np
from typing import Tuple, Dict, Any
import core.policies as policies 
from tqdm import tqdm
import gymnasium as gym
from scripts.utils.schedulers import linear_decay_schedule, exponential_decay_schedule, logarithmic_decay_schedule
from scripts.utils.utils import generate_trajectory

class ControlAgent:
    """
    A policy-based agent that selects actions based on a specified policy and updates its Q-table using incremental averages.
    """
    def __init__(
            self, 
            seed: int, 
            num_states: int, 
            num_actions: int, 
            config, 
        ) -> None:
        """
        Initialize the ControlAgent with the given parameters.

        Args:
            seed (int): The random seed for reproducibility.
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions in the environment.
            num_episodes (int): The number of episodes to run.
            policy_name (str): The name of the policy to use for action selection.
            policy_params (dict): A dictionary containing the parameters for the specified policy.
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
        self.first_visit_mc = config[config["control_algorithm"] + "_params"]["first_visit"]

        # Counters for UCB and Thompson Sampling policies
        self.state_count = np.zeros(self.num_states, dtype=np.int32)
        self.action_counts = np.zeros((self.num_states, self.num_actions), dtype=np.int32)

        # Pre-computation of discounts for the return values
        self.discounts = np.logspace(0, self.max_steps, num=self.max_steps, base=self.gamma, endpoint=False)

        # Pre-computation of learning rates for each episode (optional, can be constant)     
        self.alphas = self.init_scheduler(
            init_val=self.init_alpha, 
            min_val=self.min_alpha, 
            rate=self.alpha_decay_rate, 
            law=self.alpha_decay_law
        )  

        # Set policy-specific parameters
        # Epsilon Decay Policy
        if self.policy_name == "epsilon_greedy":
            self.init_epsilon = self.policy_params["init_epsilon"]
            self.min_epsilon = self.policy_params["min_epsilon"]
            self.epsilon_decay_rate = self.policy_params["decay_rate"]
            self.epsilon_decay_law = self.policy_params["decay_law"]
            
            # Epsilon decay schedule
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
            
            # Temperature decay schedule
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
            self.alpha = self.policy_params["alpha"]
            self.beta = self.policy_params["beta"]

        else:
            raise ValueError(f"Invalid policy name: {self.policy_name}")

        # Initialize Q-table
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)

        # Outputs placeholders
        self.v_function = np.zeros(num_states, dtype=np.float32)
        self.final_policy = np.zeros(num_states, dtype=np.int32)

    def init_scheduler(self, init_val, min_val, rate, law):
        """
        """
        if law == 'linear':
            return linear_decay_schedule(init_value=init_val, min_value=min_val, num_episodes=self.num_episodes)
        
        elif law == 'exponential':
            return exponential_decay_schedule(init_value=init_val, min_value=min_val, decay_rate=rate, num_episodes=self.num_episodes)
        
        elif law == 'logarithmic':
            return logarithmic_decay_schedule(init_value=init_val, min_value=min_val, decay_rate=rate, num_episodes=self.num_episodes)
        
        else:
            raise ValueError(f"Invalid decay law: {law}")

    def select_action(self, episode, state: int, action_space: gym.Space) -> Tuple[int, str]:
        """
        Selects an action based on the current policy and Q-table.

        Args:
            state (int): The current state of the environment.
            action_space (gym.Space): The action space of the environment, used to sample random actions.

        Returns:
            action (int): The index of the selected action.
            reason (str): The reason for selecting the action.
        """  
        q_values = self.q_table[state]

        # Use the specified policy to select an action based on the Q-table
        if self.policy_name == "epsilon_greedy":
            # Implement epsilon-greedy action selection
            return policies.epsilon_greedy_policy(q_values, self.epsilons[episode], action_space, np_random=self.np_random)
        
        elif self.policy_name == "softmax":
            # Implement softmax action selection
            return policies.softmax_policy(q_values, self.temperatures[episode], np_random=self.np_random)
        
        elif self.policy_name == "ucb":
            # Implement Upper Confidence Bound (UCB) action selection 
            return policies.ucb_policy(q_values, self.c, self.state_count[state], self.action_counts[state], np_random=self.np_random)
        
        elif self.policy_name == "thompson_sampling":
            # Implement Thompson Sampling action selection
            return policies.thompson_sampling_policy(q_values, self.alpha, self.beta, self.action_counts[state], np_random=self.np_random)
        
        else:
            raise ValueError("Unsupported policy")
    
    def run(self, env: gym.Env, Q_star, V_star):
        """
        
        """
        self.mae_history = []
        self.rewards_history = []
        self.running_average_rewards = []
        self.total_regret_history = []

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

    def run_mc_agent(self, env, Q_star, V_star):
        """
        """
        # Training loop
        cumulative_regret = 0.0
        pbar = tqdm(self.episodes, leave=True, desc="MC Control Training", unit="episode")
        for ep in pbar:
            if self.policy_name == "epsilon_greedy":
                pbar.set_postfix(epsilon=f"{self.epsilons[ep]:.2f}")
                
            elif self.policy_name == "softmax":
                pbar.set_postfix(temperature=f"{self.temperatures[ep]:.2f}")
            
            # Lambda function to select the action (policy) to pass to the trajectory generation function
            select_action_fn = lambda s: self.select_action(episode=ep, state=s, action_space=env.action_space)[0]

            # Generate trajectory
            trajectory = generate_trajectory(agent_select_action_fn=select_action_fn, env=env, max_steps=self.max_steps)

            # Calculating historical metrics
            episode_reward = sum([step[2] for step in trajectory])
            self.rewards_history.append(episode_reward)
            self.running_average_rewards.append(np.mean(self.rewards_history))

            # Cumulative Regret calculation based on the starting state of the current episode
            if V_star is not None and len(trajectory) > 0:
                initial_state = trajectory[0][0]
                episode_regret = V_star[initial_state] - episode_reward
                cumulative_regret += max(0.0, episode_regret)
            self.total_regret_history.append(cumulative_regret)

            # Visited actions-states
            visited = np.zeros((self.num_states, self.num_actions), dtype=bool)
            
            # For each timestamp of the trajectory
            for t, (state, action, _, _, _) in enumerate(trajectory):
                # Updated exploration counters for UCB and Thompson Sampling 
                self.state_count[state] += 1
                self.action_counts[state, action] += 1
                
                # MC Control: First Visit Logic
                if visited[state][action] and self.first_visit_mc:
                    continue
                visited[state][action] = True

                # Remaining steps to the end of the trajectory
                n_steps = len(trajectory[t:])

                # Computing the discounted cumulative return
                _return = np.sum(self.discounts[:n_steps] * trajectory[t:, 2])

                # Update of thr Q-table
                self.q_table[state][action] += self.alphas[ep] * (_return - self.q_table[state][action])
            
            # Computing the MAE on the Q-table
            if Q_star is not None:
                self.mae_history.append(np.mean(np.abs(Q_star - self.q_table)))
            else:
                self.mae_history.append(0.0)

        # Compute the V-function and thr policy corrisponding to the final Q-table
        self.v_function = np.max(self.q_table, axis=1)
        self.final_policy = np.argmax(self.q_table, axis=1)

    def run_sarsa_agent(self, env, Q_star, V_star):
        """
        """
        # Lambda function to select the action (policy) to pass to the trajectory generation function
        select_action_fn = lambda s: self.select_action(episode=ep, state=s, action_space=env.action_space)[0]
        
        # Training loop
        cumulative_regret = 0.0
        pbar = tqdm(self.episodes, leave=True, desc="MC Control Training", unit="episode")
        for ep in pbar:
            if self.policy_name == "epsilon_greedy":
                pbar.set_postfix(epsilon=f"{self.epsilons[ep]:.2f}")
                
            elif self.policy_name == "softmax":
                pbar.set_postfix(temperature=f"{self.temperatures[ep]:.2f}")

            state, info = env.reset()
            action = select_action_fn(state)

            # Iterate until the end of the episode
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = select_action_fn(next_state)

                # SARSA update of the Q-table
                td_target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alphas[ep] * td_error

                # Update the action and the state for the next iteration
                state, action = next_state, next_action

        # Compute the V-function and thr policy corrisponding to the final Q-table
        self.v_function = np.max(self.q_table, axis=1)
        self.final_policy = np.argmax(self.q_table, axis=1)

    def run_q_learning_agent(self, env, Q_star, V_star):
        """
        """
        pass

    def run_double_q_learning_agent(self, env, Q_star, V_star):
        """
        """
        pass