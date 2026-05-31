import numpy as np

from scripts.core.policies import random_policy
from scripts.utils.utils import linear_decay_schedule, exponential_decay_schedule

class ValuePredictionAgent:
    def __init__(self, seed, num_episodes, num_states, num_actions, algorithm_params):
        """
        Agent for value prediction using TD(0) or TD(lambda). This agent focuses on learning the value function (V-table) rather than a policy (Q-table).

        Args:
            seed (int): The random seed for reproducibility.
            num_episodes (int): The number of episodes to run.
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions in the environment.
            algorithm_params (dict): A dictionary containing the parameters for the specified algorithm.
        """
        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Set the agent's parameters
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

        # Pre-computation of learning rates for each episode (optional, can be constant)
        if algorithm_params["decay_law"] == "linear":
            self.alpha = linear_decay_schedule(init_value=self.init_alpha, min_value=self.min_alpha, decay_rate=self.decay_rate, num_episodes=self.num_episodes)
        
        elif algorithm_params["decay_law"] == "exponential":
            self.alpha = exponential_decay_schedule(init_value=self.init_alpha, min_value=self.min_alpha, decay_rate=self.decay_rate, num_episodes=self.num_episodes)
        
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

    def select_action(self, state):
        """
        We use a random policy for action selection since our focus is on value prediction, not control.

        Args:
            state (int): The current state of the environment.

        Returns:
            action (int): The index of the selected action.
        """
        return random_policy(state)

    def reset_traces(self):
        """
        The Eligibility Traces must be resetted at the beginning of each episode.
        """
        self.eligibility_traces = np.zeros(self.num_states)

    def update(self, state, reward, next_state, done):
        """
        Smista l'aggiornamento all'algoritmo corretto impostato nello YAML.
        """
        if self.algorithm_name == "td_zero":
            self._update_td_zero(state, reward, next_state, done)

        elif self.algorithm_name == "td_lambda":
            self._update_td_lambda(state, reward, next_state, done)

        else:
            raise ValueError(f"Algoritmo '{self.algorithm_name}' non supportato per la prediction.")

    def _update_td_zero(self, episode, state, reward, next_state, done):
        """
        Implementation of pure TD(0): local update O(1).

        Args:
            episode (int): The current episode number.
            state (int): The current state of the environment.
            reward (float): The reward received for taking the current action in the current state.
            next_state (int): The next state the agent will be in after taking the current action.
            done (bool): Whether the episode has ended.
        """
        # Compute the TD target: reward + discounted value of the next state. If the current state is terminal, we don't consider the value of the next state.
        td_target = reward + self.gamma * self.v_table[next_state] * (1 - done)  
        
        # Calculation of TD Error: Target - Current Estimate
        td_error = td_target - self.v_table[state]
        
        # Update the value of the current state based on the TD error
        self.v_table[state] += self.alpha[episode] * td_error

    def _update_td_lambda(self, episode, state, reward, next_state, done):
        """
        Implementation of Backward TD(lambda): global update O(S).

        Args:
            episode (int): The current episode number.
            state (int): The current state of the environment.
            reward (float): The reward received for taking the current action in the current state.
            next_state (int): The next state the agent will be in after taking the current action.
            done (bool): Whether the episode has ended.
        """
        v_next = 0.0 if done else self.v_table[next_state]
        td_error = reward + self.gamma * v_next - self.v_table[state]

        # Accumulating Trace: aumenta la traccia dello stato appena visitato
        self.eligibility_traces[state] += 1.0

        # Global Update: applichiamo l'errore corrente a TUTTI gli stati
        # in proporzione a quanto tempo fa sono stati visitati (eligibility_traces)
        for s in range(self.num_states):
            self.v_table[s] += self.alpha[episode] * td_error * self.eligibility_traces[s]
            
            # Fai decadere la traccia per il prossimo step temporale
            self.eligibility_traces[s] *= self.gamma * self._lambda