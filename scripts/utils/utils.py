import gymnasium as gym

import yaml
import numpy as np

def read_config_params(file_path):
    """
    Reads the parameters for the FrozenLake environment from a YAML file.

    Args:
        file_path (str): The path to the YAML file containing the parameters.

    Returns:
        params (dict): A dictionary containing the parameters for the FrozenLake environment.
    """
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def compute_alpha_beta(Q, ts_alpha, ts_beta, N, unimodal=True):
    """
    Computes the alpha and beta parameters for the Beta distribution based on the Q-values and action counts.

    Args:
        Q (numpy.ndarray): The Q-values for each action.
        ts_alpha (float): The alpha parameter of the Thompson Sampling.
        ts_beta (float): The beta parameter of the Thompson Sampling.
        N (numpy.ndarray): The number of times each action has been taken.

    Returns:
        alpha (numpy.ndarray): The alpha parameters for each action.
        beta (numpy.ndarray): The beta parameters for each action.
    """
    # We set the mean of the Beta distribution to be the Q-values and the variance to be a function of the alpha and beta parameters, which are in turn influenced by the action counts.
    eps = 1e-6
    mean = np.clip(Q, eps, 1 - eps)
    std_dev = ts_alpha / (ts_beta + np.sqrt(N))
    max_std = np.sqrt(mean * (1 - mean)) * 0.99
    std_dev = np.minimum(std_dev, max_std)

    # Compute alpha and beta for the specified mean and variance based on the Q-values and action counts
    common = (mean - mean ** 2 - std_dev ** 2) / std_dev ** 2
    alpha = mean * common
    beta = (1 - mean) * common

    # Ensure that alpha and beta are greater than 1 to make them Beta distribution unimodal
    if unimodal:
        alpha = 1 + np.log(1 + np.exp(alpha))
        beta = 1 + np.log(1 + np.exp(beta))

    return alpha, beta

def generate_trajectory(agent_select_action_fn, env: gym.Env, max_steps: int, complete_trajectory: bool = False):
    """
    Generates a single trajectory up to max_steps using pre-allocated numpy arrays.
    
    Args:
        agent_select_action_fn (callable): Lambda or function to select an action given a state.
        env (gym.Env): The Gymnasium environment.
        max_steps (int): Maximum duration allowed for a single simulation loop.
        complete_trajectory (bool): 
            If False, returns the experiences accumulated up to max_steps even if not terminated.
            If True, resets and retries until a successful termination (done=True) occurs within max_steps.

    Returns:
        list of tuple: A lightweight list containing tuples of (state, action, reward, next_state, done)
                       for each transition in the recorded trajectory.
    """    
    # Memory pre-allocation (Executed only once at the beginning)
    states = np.zeros(max_steps, dtype=np.int32)
    actions = np.zeros(max_steps, dtype=np.int32)
    rewards = np.zeros(max_steps, dtype=np.float32)
    next_states = np.zeros(max_steps, dtype=np.int32)
    dones = np.zeros(max_steps, dtype=np.bool_)
    steps_taken = max_steps 
    
    # Main simulation loop
    while True:
        state, _ = env.reset()
        success = False

        for t in range(max_steps):
            # Action selection and execution
            action = agent_select_action_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 

            # Direct memory write by index (overwrites previous failed attempts, if any)
            states[t] = state
            actions[t] = action
            rewards[t] = reward
            next_states[t] = next_state
            dones[t] = done

            if done:
                steps_taken = t + 1
                success = True
                break

            state = next_state

        # Dual exit condition check
        # If complete_trajectory is False, exit immediately (timeout is acceptable)
        # If complete_trajectory is True, exit only if the episode reached a valid 'done' status
        if not complete_trajectory or success:
            break

    # Final safety check on the trajectory length
    if steps_taken > max_steps:
        raise RuntimeError("Invalid trajectory length")

    # Return the lightweight list of standard tuples
    return [
        (states[i], actions[i], rewards[i], next_states[i], dones[i]) 
        for i in range(steps_taken)
    ]

