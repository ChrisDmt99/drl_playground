import numpy as np
from typing import Tuple, Any
from utils.utils import compute_alpha_beta

def policy_evaluation(pi: np.ndarray, P: dict, gamma: float, theta: float) -> np.ndarray:
    """
    Iterative Policy Evaluation (Gauss-Seidel version). 
    Computes V^pi for a fixed policy pi by iteratively solving the Bellman expectation equation.
    Formula: V(s) = Sum_{s'} P(s'|s,pi(s)) * [R(s,pi(s),s') + gamma * V(s')].

    Args:
        pi: policy array mapping state -> action
        P: environment transition model (env.unwrapped.P)
        gamma: discount factor
        theta: convergence threshold

    Returns:
        V: state-value function for policy pi
    """
    num_states = len(P)
    V = np.zeros(num_states, dtype=np.float64)

    while True:
        delta = 0.0
        for s in range(num_states):
            v_old = V[s]
            action = int(pi[s])
            v_new = 0.0
            for prob, next_state, reward, done in P[s][action]:
                v_new += prob * (reward + gamma * V[next_state] * (not done))

            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            break

    return V

def policy_improvement(V: np.ndarray, P: dict, gamma: float) -> np.ndarray:
    """
    Policy Improvement step: Given a value function V, compute the new policy pi' 
    that is greedy with respect to V using a one-step look-ahead.
    Formula: pi'(s) = argmax_a Sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')].

    Args:
        V (numpy.ndarray): The value function used to compute the new policy.
        P (dict): The transition probabilities and rewards of the environment.
        gamma (float): The discount factor for future rewards.

    Returns:
        new_pi (numpy.ndarray): The improved policy array that is greedy with respect to V.
    """
    # Initialize Q-table local buffer to store action values for each state
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    # The new policy selects the action index with the highest Q-value for each state
    new_pi = np.argmax(Q, axis=1)

    return new_pi

def compute_optimal_policy(env: Any, gamma: float, theta: float) -> np.ndarray:
    """
    Helper function to compute the optimal policy of the environment using Policy Iteration (Planning).
    Alternates between Policy Evaluation and Policy Improvement until the policy stabilizes.

    Args:
        env (gym.Env): The environment for which to compute the optimal policy.
        gamma (float): The discount factor for future rewards.
        theta (float): A small threshold for determining convergence.

    Returns:
        pi (numpy.ndarray): The computed optimal policy as an array of action indices.
    """
    num_states = env.observation_space.n
    P = env.unwrapped.P
    
    # Initialize policy deterministically (all states map to action index 0)
    pi = np.zeros(num_states, dtype=int)

    while True:
        old_pi = pi.copy()
        
        # Step 1: Evaluate current policy to find its true value function V^pi
        V = policy_evaluation(pi, P, gamma, theta)
        
        # Step 2: Improve policy greedily based on the newly computed V^pi
        pi = policy_improvement(V, P, gamma)
        
        # Convergence check: if the policy mapping does not change, pi* is found
        if np.array_equal(old_pi, pi):
            break

    return pi

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
    if np.isnan(q_values).any():
        raise RuntimeError("Found NaN in Q-values.")

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

def thompson_sampling_policy(q_values: np.ndarray, ts_alpha: float, ts_beta: float, action_counts: np.ndarray, np_random: np.random.Generator, unimodal: bool = True) -> Tuple[int, str]:
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
    besta_dist_alpha, besta_dist_beta = compute_alpha_beta(Q=q_values, ts_alpha=ts_alpha, ts_beta=ts_beta, N=action_counts, unimodal=unimodal)
    sampled_q = np_random.beta(besta_dist_alpha, besta_dist_beta)      
    best_actions = np.where(sampled_q == np.max(sampled_q))[0]
    best_action = np_random.choice(best_actions)
    return int(best_action), "Thompson Sampling Action"