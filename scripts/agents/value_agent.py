import numpy as np
from typing import Tuple, Dict, Any

from scripts.utils.schedulers import linear_decay_schedule, exponential_decay_schedule, logarithmic_decay_schedule
from scripts.core.value_functions import update_td_zero, update_td_lambda

class ValuePredictionAgent:
    def __init__(self, seed: int, action_space: Any, num_episodes: int, num_states: int, num_actions: int, algorithm_params: Dict[str, Any], policy: list = None) -> None:
        """
        Agent for value prediction using TD(0) or TD(lambda). This agent focuses on learning the value function (V-table) rather than a policy (Q-table).

        Args:
            seed (int): The random seed for reproducibility.
            action_space (gym.Space): The action space of the environment, used for action selection.
            num_episodes (int): The number of episodes to run.
            num_states (int): The number of states in the environment.
            num_actions (int): The number of actions in the environment.
            algorithm_params (dict): A dictionary containing the parameters for the specified algorithm.
            policy (list): A fixed policy for action selection. If None, a random policy will be used. Defaults to None.
        """
        # Set the random seed for reproducibility
        self.seed = seed
        self.np_random = np.random.default_rng(seed)

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
        self.algorithm_params = algorithm_params

        # Here we use a V-Table [num_states] instead of a Q-Table [num_states, num_actions]
        self.v_table = np.zeros(self.num_states, dtype=np.float64)        

        # To track the value function over episodes for visualization purposes
        self.v_track = np.zeros((self.num_episodes, self.num_states), dtype=np.float64)

        # Set the policy for action selection (in this case, we use a fixed policy)
        self.policy = np.array(policy) if policy is not None else self.np_random.integers(0, self.num_actions, size=self.num_states)
        self.reason = "target Evaluation Policy" if policy is not None else "Random Policy"

        # Pre-computation of learning rates for each episode (optional, can be constant)        
        if self.algorithm_params['decay_law'] == 'linear':
            self.alphas = linear_decay_schedule(
                init_value=self.init_alpha, 
                min_value=self.min_alpha, 
                decay_steps=self.algorithm_params["decay_steps"],
                num_episodes=num_episodes
            )
        
        elif self.algorithm_params['decay_law'] == 'exponential':
            self.alphas = exponential_decay_schedule(
                init_value=self.init_alpha, 
                min_value=self.min_alpha, 
                decay_steps=self.algorithm_params["decay_steps"],
                decay_rate=self.algorithm_params["decay_rate"],
                num_episodes=num_episodes
            )

        elif self.algorithm_params['decay_law'] == 'logarithmic':
            self.alphas = logarithmic_decay_schedule(
                init_value=self.init_alpha, 
                min_value=self.min_alpha, 
                decay_steps=self.algorithm_params["decay_steps"],
                decay_rate=self.algorithm_params["decay_rate"],
                num_episodes=num_episodes
            )
        else:
            raise ValueError(f"Invalid decay law: {self.algorithm_params['decay_law']}")

        # Temporal Difference (TD) learning with eligibility traces (TD-lambda) requires maintaining a table of eligibility traces for each state.
        if self.algorithm_name == "td_zero":
            self.eligibility_traces = None  # Not used in TD(0)

        # Temporal Difference (TD) learning with eligibility traces (TD-lambda) requires maintaining a table of eligibility traces for each state.
        elif self.algorithm_name == "td_lambda":
            # Table of Eligibility Traces (used only in TD-lambda)
            self.reset_traces()

        else:
            raise ValueError(f"Algorithm '{self.algorithm_name}' not supported for value prediction.")

    def select_action(self, state: int) -> Tuple[int, str]:
        """
        Control Feature: Evaluates the expected value of each action using V(s') and transitions model.
        Formula: arg_max_a Sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')].

        Args:
            state (int): The current state of the environment.

        Returns:
            action (int): The index of the selected action.
            reason (str): The reason for selecting the action.
        """              
        return int(self.policy[state]), self.reason

    def reset_traces(self) -> None:
        """
        The Eligibility Traces must be resetted at the beginning of each episode.
        """
        self.eligibility_traces = np.zeros(self.num_states, dtype=np.float64)

    def update_value_function(self, episode: int, reward: float, state: int, next_state: int, done: bool) -> None:
        """
        This method can be used to perform any necessary updates to the value function at the end of each episode, such as decaying the learning rate or resetting eligibility traces.

        Args:
            episode (int): The current episode number.
            reward (float): The reward received for taking the action in the current state.
            state (int): The current state of the environment.
            next_state (int): The next state of the environment after taking the action.
            done (bool): Whether the episode has ended after taking the action.
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
    
    def end_of_episode(self, episode: int) -> None:
        """
        This method can be used to perform any necessary updates to the value function at the end of each episode, such as decaying the learning rate or resetting eligibility traces.
        
        Args:
            episode (int): The current episode number. 
        """
        self.v_track[episode] = self.v_table.copy()  # Store the value function at the end of each episode for tracking purposes
