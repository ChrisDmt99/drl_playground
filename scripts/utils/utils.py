import yaml
import numpy as np

def read_frozen_lake_params(file_path):
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

def linear_decay_schedule(init_value, min_value, decay_rate, num_episodes):
    """
    Creates a linear decay schedule for a learning rate or other hyperparameter.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        decay_rate (float): Not used in linear decay (kept for API consistency).
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    episodes = np.arange(num_episodes)
    schedule = np.maximum(min_value, init_value - (init_value - min_value) * episodes / num_episodes)
    return schedule

def exponential_decay_schedule(init_value, min_value, decay_rate, num_episodes):
    """
    Creates an exponential decay schedule for a learning rate or other hyperparameter.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        decay_rate (float): Exponential decay rate (0 < decay_rate < 1).
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    episodes = np.arange(num_episodes)
    schedule = np.maximum(min_value, min_value + (init_value - min_value) * (decay_rate ** (episodes / num_episodes)))
    return schedule

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

