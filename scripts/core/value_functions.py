import numpy as np
from typing import Tuple, Any

def compute_optimal_v_function(env: Any, gamma: float, theta: float) -> np.ndarray:
    """
    Helper function to compute the exact V* function of the environment using Value Iteration (Planning) offline.
    Formula (Bellman optimality equation): V*(s) = max_a Sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V*(s')].

    Args:
        env (gym.Env): The environment for which to compute V*.
        gamma (float): The discount factor for future rewards.
        theta (float): A small threshold for determining convergence.

    Returns:
        V (numpy.ndarray): The computed optimal value function V* for each state.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    P = env.unwrapped.P
    V = np.zeros(num_states, dtype=np.float64)

    while True:
        Q = np.zeros((num_states, num_actions), dtype=np.float64)
        for s in range(num_states):
            for a in range(num_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)

    return V

def update_td_zero(
        v_table: np.ndarray, 
        alphas: np.ndarray, 
        episode: int, 
        state: int, 
        next_state: int, 
        reward: float, 
        gamma: float, 
        done: bool
    ) -> np.ndarray:
    """
    Implementation of pure TD(0): local update O(1).

    Args:
        v_table (numpy.ndarray): The value table to update.
        alphas (numpy.ndarray): The learning rates for each episode.
        episode (int): The current episode number.
        state (int): The current state of the environment.
        next_state (int): The next state the agent will be in after taking the current action.
        reward (float): The reward received for taking the current action in the current state.
        gamma (float): The discount factor for future rewards.
        done (bool): Whether the episode has ended.

    Returns:
        v_table (numpy.ndarray): The updated value table after applying the TD(0) update.
    """
    # Compute the TD target: reward + discounted value of the next state. If the current state is terminal, we don't consider the value of the next state.
    td_target = reward + gamma * v_table[next_state] * (1 - done)  

    # Calculation of TD Error: Target - Current Estimate
    td_error = td_target - v_table[state]

    # Update the value of the current state based on the TD error
    v_table[state] += alphas[episode] * td_error

    return v_table

def update_td_lambda(
    v_table: np.ndarray, 
    eligibility_traces: np.ndarray, 
    alphas: np.ndarray, 
    episode: int, 
    state: int, next_state: int, 
    reward: float, gamma: float, 
    lambda_: float, 
    done: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of Backward TD(lambda): global update O(S).

    Args:
        v_table (numpy.ndarray): The value table to update.
        eligibility_traces (numpy.ndarray): The eligibility traces for each state.
        alphas (numpy.ndarray): The learning rates for each episode.
        episode (int): The current episode number.
        state (int): The current state of the environment.
        reward (float): The reward received for taking the current action in the current state.
        next_state (int): The next state the agent will be in after taking the current action.
        lambda_ (float): The trace decay parameter for TD(lambda).
        done (bool): Whether the episode has ended.

    Returns:
        v_table (numpy.ndarray): The updated value table after applying the TD(lambda) update.
        eligibility_traces (numpy.ndarray): The updated eligibility traces after applying the TD(lambda) update.
    """
    # Compute the TD target: reward + discounted value of the next state. If the current state is terminal, we don't consider the value of the next state.
    td_target = reward + gamma * v_table[next_state] * (1.0 - float(done))  

    # Calculation of TD Error: Target - Current Estimate
    td_error = td_target - v_table[state]

    # Update the eligibility trace for the current state (incrementing it by 1 since we just visited it)
    eligibility_traces[state] += 1.0

    # Update the value table based on the TD error and eligibility traces
    v_table += alphas[episode] * td_error * eligibility_traces

    # Decay the eligibility traces for all states (this is done after the update to ensure that the current state gets updated with the full trace value)
    if done:
        eligibility_traces.fill(0.0)
    else:
        eligibility_traces *= gamma * lambda_

    return v_table, eligibility_traces