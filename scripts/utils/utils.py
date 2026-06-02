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

def get_terminal_states_mask(num_states: int):
    """
    Helper function to identify Holes and Goal indices for 4x4 or 8x8 FrozenLake.
    Returns a 2D boolean mask where False indicates a terminal state.
    """
    side = int(np.sqrt(num_states))
    mask = np.ones((side, side), dtype=bool)
    
    if side == 4:
        # Indices: 5, 7, 11, 12 (Holes) and 15 (Goal)
        holes_goal = [5, 7, 11, 12, 15]
    elif side == 8:
        # Standard indices for 8x8 FrozenLake Holes and Goal
        holes_goal = [19, 29, 35, 41, 42, 46, 49, 54, 59, 63]
    else:
        holes_goal = [num_states - 1] # Fallback solo Goal
        
    for idx in holes_goal:
        r, c = divmod(idx, side)
        mask[r, c] = False
    return mask, side