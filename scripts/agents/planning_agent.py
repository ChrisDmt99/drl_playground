import numpy as np
from typing import Any, Tuple

from core.value_functions import compute_optimal_v_function
from core.policies import compute_optimal_policy
from core.q_functions import compute_optimal_q_function

class PlanningAgent:
    """
    An offline Planning Agent that uses Dynamic Programming (Bellman Optimality Equations)
    to compute the optimal V*, Q*, and pi* functions directly from the environment model.
    Serves as the ultimate Ground Truth generator for Reinforcement Learning agents.
    """
    def __init__(self, env: Any, gamma: float, theta: float):
        """
        Initializes the PlanningAgent with the environment and planning parameters.

        Args:
            env (gym.Env): The OpenAI Gym environment.
            gamma (float): The discount factor for future rewards.
            theta (float): A small threshold for determining convergence.        
        """
        # Store environment and planning parameters
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # Extract dimensions and transition dynamics
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.P = env.unwrapped.P
        
        # Storage for computed ground truths
        self.V_star = None
        self.Q_star = None
        self.pi_star = None

    def run_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes all planning algorithms and caches the results.

        Returns:
            V_star (numpy.ndarray): The computed optimal value function V*.
            Q_star (numpy.ndarray): The computed optimal action-value function Q*.
            pi_star (numpy.ndarray): The computed optimal policy.
        """
        # Debug
        print("[Planning Engine] Calculating complete Ground Truth mathematical solutions...")

        # Storage for computed ground truths
        self.V_star = compute_optimal_v_function(self.env, gamma=self.gamma, theta=self.theta)
        self.Q_star = compute_optimal_q_function(self.env, gamma=self.gamma, theta=self.theta)
        self.pi_star = compute_optimal_policy(self.env, gamma=self.gamma, theta=self.theta)
        
        # Debug
        print("[Planning Engine] V*, Q*, and Policy computed successfully.")
        
        return self.V_star, self.Q_star, self.pi_star