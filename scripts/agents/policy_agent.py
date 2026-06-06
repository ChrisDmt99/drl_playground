import numpy as np
from typing import Tuple, Dict, Any
import core.policies as policies 
import gymnasium as gym
from scripts.utils.schedulers import linear_decay_schedule, exponential_decay_schedule, logarithmic_decay_schedule

class PolicyAgent:
    """
    A policy-based agent that selects actions based on a specified policy and updates its Q-table using incremental averages.
    """
    def __init__(self, seed: int, num_states: int, num_actions: int, num_episodes: int, policy_name: str, policy_params: Dict[str, Any]) -> None:
        """
        Initialize the PolicyAgent with the given parameters.

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
        self.num_episodes = num_episodes
        self.policy_name = policy_name
        self.policy_params = policy_params

        # Start from zeros: the agent will learn the real values incrementally
        self.q_table = np.zeros((num_states, num_actions))

        # Core counters for learning models (moved outside specific ifs to be globally accessible)
        self.state_count = np.zeros(self.num_states)
        self.action_counts = np.zeros((self.num_states, self.num_actions))

        # Set policy-specific parameters
        if self.policy_name == "epsilon_greedy":
            self.init_epsilon = policy_params.get("init_epsilon", 1.0)
            self.min_epsilon = policy_params.get("min_epsilon", 0.01)
            self.decay_rate = policy_params.get("decay_rate", 0.9) 
            self.epsilon = self.init_epsilon 
            
            # Epsilon decay schedule
            if self.policy_params['decay_law'] == 'linear':
                self.epsilons = linear_decay_schedule(
                    init_value=self.init_epsilon, 
                    min_value=self.min_epsilon, 
                    num_episodes=num_episodes
                )
            
            elif self.policy_params['decay_law'] == 'exponential':
                self.epsilons = exponential_decay_schedule(
                    init_value=self.init_epsilon, 
                    min_value=self.min_epsilon, 
                    decay_rate=self.policy_params["decay_rate"],
                    num_episodes=num_episodes
                )

            elif self.policy_params['decay_law'] == 'logarithmic':
                self.epsilons = logarithmic_decay_schedule(
                    init_value=self.init_epsilon, 
                    min_value=self.min_epsilon, 
                    decay_rate=self.policy_params["decay_rate"],
                    num_episodes=num_episodes
                )
            else:
                raise ValueError(f"Invalid decay law: {self.policy_params['decay_law']}")

        elif self.policy_name == "softmax":
            self.init_temperature = policy_params.get("init_temperature", 1.0)
            self.min_temperature = policy_params.get("min_temperature", 0.01)
            self.decay_rate = policy_params.get("decay_rate", 0.9)
            self.temperature = self.init_temperature  

            # Temperature decay schedule
            if self.policy_params['decay_law'] == 'linear':
                self.temperatures = linear_decay_schedule(
                    init_value=self.init_temperature, 
                    min_value=self.min_temperature, 
                    num_episodes=num_episodes, 
                )
            
            elif self.policy_params['decay_law'] == 'exponential':
                self.temperatures = exponential_decay_schedule(
                    init_value=self.init_temperature, 
                    min_value=self.min_temperature, 
                    decay_rate=self.policy_params["decay_rate"],
                    num_episodes=num_episodes
                )

            elif self.policy_params['decay_law'] == 'logarithmic':
                self.temperatures = logarithmic_decay_schedule(
                    init_value=self.init_temperature, 
                    min_value=self.min_temperature, 
                    decay_rate=self.policy_params["decay_rate"],
                    num_episodes=num_episodes
                )
            else:
                raise ValueError(f"Invalid decay law: {self.policy_params['decay_law']}")

        elif self.policy_name == "ucb":
            self.c = policy_params.get("c", 1.0)

        elif self.policy_name == "thompson_sampling":
            self.alpha = policy_params.get("alpha", 1.0)
            self.beta = policy_params.get("beta", 1.0)

        else:
            raise ValueError(f"Invalid policy name: {policy_name}")

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
        
    def update(self, state: int, action: int, reward: float) -> None:
        """
        Updates the Q-table using the incremental average formula. Tracks visits and sample averages for each state-action pair dynamically.

        Args:
            state (int): The current state.
            action (int): The chosen action.
            reward (float): The reward received for taking the action in the current state.
        """
        # Increment exploration counters
        self.state_count[state] += 1
        self.action_counts[state][action] += 1
        
        # Compute dynamic step size for the sample average (alpha = 1 / N(s, a)).
        # This ensures that each reward contributes equally to the mean,
        # making early updates large and late updates progressively more stable.        
        step_size = 1.0 / self.action_counts[state][action]
        
        # Dynamic Q-table update rule
        # NewEstimate = OldEstimate + StepSize * [Target - OldEstimate]
        self.q_table[state][action] += step_size * (reward - self.q_table[state][action])