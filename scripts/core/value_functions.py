

def update_td_zero(v_table, alphas, episode, state, next_state, reward, gamma, done):
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

def update_td_lambda(v_table, eligibility_traces, alphas, episode, state, next_state, reward, gamma, lambda_, done):
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
    td_target = reward + gamma * v_table[next_state] * (1 - done)  

    # Calculation of TD Error: Target - Current Estimate
    td_error = td_target - v_table[state]

    # Update the eligibility trace for the current state (incrementing it by 1 since we just visited it)
    eligibility_traces[state] += 1

    # Update the value table based on the TD error and eligibility traces
    v_table += alphas[episode] * td_error * eligibility_traces

    # Decay the eligibility traces for all states (this is done after the update to ensure that the current state gets updated with the full trace value)
    eligibility_traces *= gamma * lambda_

    return v_table, eligibility_traces