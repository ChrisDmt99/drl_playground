import numpy as np
from typing import Union

def linear_decay_schedule(init_value: float, min_value: float, decay_steps: int, num_episodes: int) -> np.ndarray:
    """
    Creates a linear decay schedule for a learning rate or other hyperparameter.
    The parameter starts from init_value at epoch=0 and decays linearly until it reaches min_value at epoch=decay_steps.
    After that point, the parameter remains constant at min_value.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        decay_steps (int): Number of episodes over which the parameter should decay.
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    episodes = np.arange(num_episodes)
    
    # Calculate exactly at which episode the decay should stop (e.g., episode 750 out of 1000)
    if decay_steps <= 1:
        decay_steps = 1
    elif decay_steps > num_episodes:
        decay_steps = num_episodes

    # Linear interpolation applied ONLY up to the decay_steps point
    # After that point, np.maximum will seamlessly clip everything to min_value
    schedule = init_value - (init_value - min_value) * episodes / (decay_steps - 1)
    schedule = np.maximum(min_value, schedule)
    
    return schedule

def exponential_decay_schedule(init_value: float, min_value: float, decay_rate: float, decay_steps: int, num_episodes: int) -> np.ndarray:
    """
    Creates an exponential decay schedule for a learning rate or other hyperparameter.
    The parameter starts from init_value at epoch=0 and decays exponentially until epoch=decay_steps, and then remains constant at min_value.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        decay_rate (float): Exponential decay rate (typically 0.9 <= decay_rate < 1.0).
        decay_steps (int): Number of episodes over which the parameter should decay.
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    # Defensive checks for decay_steps boundaries
    if decay_steps <= 1:
        decay_steps = 1
    elif decay_steps > num_episodes:
        decay_steps = num_episodes

    # Generate exponential curve only for the active decay period
    decay_episodes = np.arange(decay_steps)
    decay_period = init_value * (decay_rate ** decay_episodes)
    
    # Clip the active period to guarantee it doesn't fall below min_value early
    decay_period = np.maximum(min_value, decay_period)
    
    # Create the flat padding array for the remaining episodes
    remaining_steps = num_episodes - decay_steps
    flat_period = np.full(shape=remaining_steps, fill_value=min_value)
    
    # Concatenate into the final schedule
    schedule = np.concatenate([decay_period, flat_period])
    
    return schedule

def logarithmic_decay_schedule(init_value: float, min_value: float, decay_rate: float, decay_steps: int, num_episodes: int) -> np.ndarray:
    """
    Creates a logarithmic decay schedule based on classic cooling schedules.
    The parameter starts from init_value at epoch=0 and decays logarithmically until epoch=decay_steps, and then remains constant at min_value.

    Args:
        init_value (float): Initial value of the parameter.
        min_value (float): Minimum value the parameter should decay to.
        decay_rate (float): Logarithmic scaling rate factor (decay_rate > 0).
        decay_steps (int): Number of episodes over which the parameter should decay.
        num_episodes (int): Total number of episodes for the decay schedule.

    Returns:
        numpy.ndarray: Array of decayed values for each episode.
    """
    # Defensive checks for decay_steps boundaries
    if decay_steps <= 1:
        decay_steps = 1
    elif decay_steps > num_episodes:
        decay_steps = num_episodes

    # 1. Generate logarithmic curve only for the active decay period
    decay_episodes = np.arange(decay_steps)
    decay_period = init_value / (1.0 + decay_rate * np.log(1.0 + decay_episodes))
    
    # Clip the active period to guarantee it doesn't fall below min_value early
    decay_period = np.maximum(min_value, decay_period)
    
    # 2. Create the flat padding array for the remaining episodes
    remaining_steps = num_episodes - decay_steps
    flat_period = np.full(shape=remaining_steps, fill_value=min_value)
    
    # 3. Concatenate into the final schedule
    schedule = np.concatenate([decay_period, flat_period])
    
    return schedule