from turtle import done

import numpy as np

from scripts.core.policies import random_policy
from scripts.utils.utils import linear_decay_schedule, exponential_decay_schedule
from scripts.core.value_functions import update_td_zero, update_td_lambda

class ValuePredictionAgent:
    def __init__(self, seed, action_space, num_episodes, num_states, num_actions, algorithm_params):
        """
        Agent for value prediction using TD(0) or TD(lambda). This agent focuses on learning the value function (V-table) rather than a policy (Q-table).

        Args:
            seed (int): The random seed for reproducibility.
            action_space (gym.Space): The action space of the environment, used for action selection.
            num_episodes (int): The number of episodes to run.
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions in the environment.
            algorithm_params (dict): A dictionary containing the parameters for the specified algorithm.
        """
        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Set the agent's parameters
        self.action_space = action_space
        self.num_episodes = num_episodes
        self.num_states = num_states
        self.num_actions = num_actions
        self.algorithm_name = algorithm_params["algorithm_name"]
        self.init_alpha = algorithm_params["init_alpha"]
        self.min_alpha = algorithm_params["min_alpha"]
        self.decay_rate = algorithm_params["decay_rate"]
        self.gamma = algorithm_params["gamma"]
        self._lambda = algorithm_params["lambda"]  # _lambda for not confusing with Python's lambda keyword

        # Here we use a V-Table [num_states] instead of a Q-Table [num_states, num_actions]
        self.v_table = np.zeros(self.num_states)

        # To track the value function over episodes for visualization purposes
        self.v_track = np.zeros((self.num_episodes, self.num_states))  

        # Pre-computation of learning rates for each episode (optional, can be constant)
        if algorithm_params["decay_law"] == "linear":
            self.alphas = linear_decay_schedule(init_value=self.init_alpha, min_value=self.min_alpha, decay_rate=self.decay_rate, num_episodes=self.num_episodes)
        
        elif algorithm_params["decay_law"] == "exponential":
            self.alphas = exponential_decay_schedule(init_value=self.init_alpha, min_value=self.min_alpha, decay_rate=self.decay_rate, num_episodes=self.num_episodes)
        
        else:
            raise ValueError(f"Decay law '{algorithm_params['decay_law']}' not supported.")

        # Temporal Difference (TD) learning with eligibility traces (TD-lambda) requires maintaining a table of eligibility traces for each state.
        if self.algorithm_name == "td_zero":
            self.eligibility_traces = None  # Not used in TD(0)

        # Temporal Difference (TD) learning with eligibility traces (TD-lambda) requires maintaining a table of eligibility traces for each state.
        elif self.algorithm_name == "td_lambda":
            # Table of Eligibility Traces (used only in TD-lambda)
            self.reset_traces()

        else:
            raise ValueError(f"Algorithm '{self.algorithm_name}' not supported for value prediction.")

    def select_action(self):
        """
        We use a random policy for action selection since our focus is on value prediction, not control.

        Args:
            state (int): The current state of the environment.

        Returns:
            action (int): The index of the selected action.
        """
        return random_policy(self.action_space)

    def reset_traces(self):
        """
        The Eligibility Traces must be resetted at the beginning of each episode.
        """
        self.eligibility_traces = np.zeros(self.num_states)

    def update_value_function(self, episode, reward, state, next_state, done):
        """
        This method can be used to perform any necessary updates to the value function at the end of each episode, such as decaying the learning rate or resetting eligibility traces.
        """
        if self.algorithm_name == "td_zero":
            self.v_table = update_td_zero(
                v_table=self.v_table, 
                alphas=self.alphas, 
                episode=episode, 
                state=state, 
                next_state=next_state, 
                reward=reward, 
                gamma=self.gamma, 
                done=done
            )

        elif self.algorithm_name == "td_lambda":
            self.v_table, self.eligibility_traces = update_td_lambda(
                v_table=self.v_table, 
                eligibility_traces=self.eligibility_traces, 
                alphas=self.alphas, 
                episode=episode, 
                state=state, 
                next_state=next_state, 
                reward=reward, 
                gamma=self.gamma, 
                lambda_=self._lambda, 
                done=done
            )

        else:
            raise ValueError(f"Algorithm '{self.algorithm_name}' not supported for value prediction.")
    
    def end_of_episode(self, episode):
        """
        This method can be used to perform any necessary updates to the value function at the end of each episode, such as decaying the learning rate or resetting eligibility traces.
        
        Args:
            episode (int): The current episode number. 
        """
        self.v_track[episode] = self.v_table.copy()  # Store the value function at the end of each episode for tracking purposes
