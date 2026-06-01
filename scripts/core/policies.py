import numpy as np
from typing import Tuple, Any
from utils.utils import compute_alpha_beta

def random_policy(action_space: Any, np_random: np.random.Generator) -> Tuple[int, str]:
    """
    Select a random action from the action space.
    
    Args:
        action_space (gym.Space): The action space of the environment, used to sample random actions.
        np_random (np.random.Generator): The random number generator.

    Returns:
        action (int): The index of the selected action.
    """
    # Exploration: choose a random action
    action = np_random.integers(0, action_space.n)
    return int(action), "Random Action"  
  
def epsilon_greedy_policy(q_values: np.ndarray, epsilon: float, action_space: Any, np_random: np.random.Generator) -> Tuple[int, str]:
    """
    Epsilon-Greedy policy selection based on the Q-values and an epsilon parameter.
    
    Args:
        q_values (numpy.ndarray): The Q-values for each action.
        epsilon (float): The exploration probability.
        action_space (gym.Space): The action space of the environment, used to sample random actions.
        np_random (np.random.Generator): The random number generator.

    Returns:
        action (int): The index of the selected action.
    """
    # Exploration: choose a random action
    if np_random.random() < epsilon:
        action = np_random.integers(0, action_space.n)
        return int(action), "Epsilon-Greedy Random Action"
    
    # Exploitation: choose the action with the highest Q-value
    else:
        best_actions = np.where(q_values == np.max(q_values))[0]
        best_action = np_random.choice(best_actions)
        return int(best_action), "Epsilon-Greedy Best Action"
    
def softmax_policy(q_values: np.ndarray, temperature: float, np_random: np.random.Generator) -> Tuple[int, str]:
    """
    Select an action using the softmax policy based on the Q-values and a temperature parameter.

    Args:
        q_values (numpy.ndarray): The Q-values for each action.
        temperature (float): The temperature parameter for the softmax function. Higher values lead to more exploration, while lower values lead to more exploitation.
        np_random (np.random.Generator): The random number generator.

    Returns:
        action (int): The index of the selected action.
    """
    q_values_scaled = q_values / (temperature)  
    exp_q = np.exp(q_values_scaled - np.max(q_values_scaled))
    probabilities = exp_q / np.sum(exp_q)
    action = np_random.choice(len(probabilities), p=probabilities)
    return int(action), "Softmax Action"

def ucb_policy(q_values: np.ndarray, c: float, state_count: int, action_counts: np.ndarray, np_random: np.random.Generator) -> Tuple[int, str]:
    """
    Upper Confidence Bound (UCB) policy selection for a multi-armed bandit problem. In a more complex environment, you would maintain separate counts and rewards
    for each action to compute the UCB values.

    Args:
        q_values (numpy.ndarray): The Q-values for each action.
        c (float): The exploration parameter for UCB.
        state_count (int): The total number of times the current state has been visited.
        action_counts (numpy.ndarray): The number of times each action has been taken from the current state.
        np_random (np.random.Generator): The random number generator.

    Returns:
        action (int): The index of the selected action.
    """
    # First, check for any unvisited actions to ensure they are explored at least once
    unvisited = np.where(action_counts == 0)[0]

    # If there are unvisited actions, select one of them randomly
    if len(unvisited) > 0:
        action = np_random.choice(unvisited)
        return int(action), "UCB Unvisited Action"
    
    # Compute UCB values for all actions
    ucb_values = q_values + c * np.sqrt(np.log(1 + state_count) / (action_counts))

    # Choose randomly among the actions with the highest UCB value to break ties
    best_ucb_actions = np.where(ucb_values == np.max(ucb_values))[0]
    action = np_random.choice(best_ucb_actions)
    return int(action), "UCB Action"

def thompson_sampling_policy(q_values: np.ndarray, ts_alpha: float, ts_beta: float, action_counts: np.ndarray, np_random: np.random.Generator) -> Tuple[int, str]:
    """
    Thompson Sampling policy selection for a Bernoulli bandit problem. In a more complex environment, you would maintain separate alpha and beta parameters 
    for each action based on observed rewards.
    np_random (np.random.Generator): The random number generator.

    Args:
        q_values (numpy.ndarray): The Q-values for each action.
        ts_alpha (float): The alpha parameter of the Thompson Sampling.
        ts_beta (float): The beta parameter of the Thompson Sampling.
        action_counts (numpy.ndarray): The number of times each action has been taken.

    Returns:
        best_action (Tuple[int, str]): The index of the selected action.
    """
    besta_dist_alpha, besta_dist_beta = compute_alpha_beta(Q=q_values, ts_alpha=ts_alpha, ts_beta=ts_beta, N=action_counts, unimodal=False)
    sampled_q = np_random.beta(besta_dist_alpha, besta_dist_beta)      
    best_actions = np.where(sampled_q == np.max(sampled_q))[0]
    best_action = np_random.choice(best_actions)
    return int(best_action), "Thompson Sampling Action"