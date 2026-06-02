import numpy as np
from typing import Any

def compute_optimal_q_function(env: Any, gamma: float, theta: float) -> np.ndarray:
    """
    Helper function to compute the exact Q* function of the environment using Q Iteration (Planning) offline.
    Formula (Bellman optimality equation): Q*(s,a) = Sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q(s',a')].

    Args:
        env (gym.Env): The environment for which to compute Q*.
        gamma (float): The discount factor for future rewards.
        theta (float): A small threshold for determining convergence.

    Returns:
        Q (numpy.ndarray): The computed optimal action-value function Q* for each state-action pair.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    P = env.unwrapped.P
    Q = np.zeros((num_states, num_actions), dtype=np.float64)

    while True:
        delta = 0
        for s in range(num_states):                
            for a in range(num_actions):
                q_old = Q[s, a]
                q_buffer = 0.0
                for prob, next_state, reward, done in P[s][a]:
                    q_next = 0.0 if done else Q[next_state, :].max()
                    q_buffer += prob * (reward + gamma * q_next)
                Q[s, a] = q_buffer
                delta = max(delta, abs(q_old - Q[s, a]))
        if delta < theta:
            break

    return Q