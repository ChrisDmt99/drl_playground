import numpy as np
from typing import Union

def linear_decay_schedule(init_value: float, min_value: float, num_episodes: int) -> np.ndarray:
    """
    Creates a linear decay schedule for a learning rate or other hyperparameter.
    The parameter starts from init_value at episode=0 and decays linearly across all 
    episodes until it reaches min_value at the final episode.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    episodes = np.arange(num_episodes)
    
    if num_episodes <= 1:
        return np.full(shape=num_episodes, fill_value=min_value)

    # Linear interpolation across the entire training duration
    schedule = init_value - (init_value - min_value) * episodes / (num_episodes - 1)
    schedule = np.maximum(min_value, schedule)
    
    return schedule

def exponential_decay_schedule(init_value: float, min_value: float, decay_rate: float, num_episodes: int) -> np.ndarray:
    """
    Creates an exponential decay schedule for a learning rate or other hyperparameter.
    The parameter starts from init_value at episode=0 and decays exponentially across
    all episodes, bound dynamically by min_value.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        decay_rate (float): Exponential decay rate (typically 0.9 <= decay_rate < 1.0).
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    episodes = np.arange(num_episodes)
    
    # Pure exponential decay curve across the entire sequence
    schedule = init_value * (decay_rate ** episodes)
    
    # Seamless clipping at min_value when the threshold is naturally crossed
    schedule = np.maximum(min_value, schedule)
    
    return schedule

def logarithmic_decay_schedule(init_value: float, min_value: float, decay_rate: float, num_episodes: int) -> np.ndarray:
    """
    Creates a logarithmic decay schedule based on classic cooling schedules.
    The parameter starts from init_value at episode=0 and decays logarithmically across
    all episodes, bound dynamically by min_value.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        decay_rate (float): Logarithmic scaling rate factor (decay_rate > 0).
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    episodes = np.arange(num_episodes)
    
    # Logarithmic cooling applied continuously across all episodes
    schedule = init_value / (1.0 + decay_rate * np.log(1.0 + episodes))
    
    # Seamless clipping at min_value when the threshold is naturally crossed
    schedule = np.maximum(min_value, schedule)
    
    return schedule