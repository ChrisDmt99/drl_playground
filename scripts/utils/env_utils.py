import numpy as np


def _unwrap_env(env):
    """Return the base Gymnasium environment."""
    return env.unwrapped if hasattr(env, "unwrapped") else env


def get_grid_shape(env) -> tuple[int, int]:

    env = _unwrap_env(env)

    env_name = type(env).__name__

    if env_name == "FrozenLakeEnv":
        return env.nrow, env.ncol

    if env_name == "CliffWalkingEnv":
        return env.shape

    raise NotImplementedError(
        f"{env_name} is not a supported grid environment."
    )


def get_terminal_states(env) -> list[int]:

    env = _unwrap_env(env)

    env_name = type(env).__name__

    if env_name == "FrozenLakeEnv":

        rows, cols = env.nrow, env.ncol

        terminals = []

        for r in range(rows):
            for c in range(cols):

                cell = env.desc[r, c]

                if isinstance(cell, bytes):
                    cell = cell.decode()

                if cell == "H":
                    terminals.append(r * cols + c)

        return terminals

    if env_name == "CliffWalkingEnv":
        return {}

    raise NotImplementedError(
        f"{env_name} is not a supported grid environment."
    )

def get_special_states(env) -> list[int]:

    env = _unwrap_env(env)

    env_name = type(env).__name__

    if env_name == "FrozenLakeEnv":
        return {}

    if env_name == "CliffWalkingEnv":

        rows, cols = env.shape

        # Goal + cliff cells
        return list(range((rows - 1) * cols + 1, rows * cols - 1))

    raise NotImplementedError(
        f"{env_name} is not a supported grid environment."
    )

def get_goal_states(env) -> list[int]:

    env = _unwrap_env(env)

    env_name = type(env).__name__

    if env_name == "FrozenLakeEnv":

        rows, cols = env.nrow, env.ncol

        goals = []

        for r in range(rows):
            for c in range(cols):

                cell = env.desc[r, c]

                if isinstance(cell, bytes):
                    cell = cell.decode()

                if cell == "G":
                    goals.append(r * cols + c)

        return goals

    if env_name == "CliffWalkingEnv":

        rows, cols = env.shape

        return [rows * cols - 1]

    raise NotImplementedError(
        f"{env_name} is not a supported grid environment."
    )


def get_action_names(env) -> list[str]:

    env = _unwrap_env(env)

    env_name = type(env).__name__

    if env_name == "FrozenLakeEnv":

        return [
            "←",
            "↓",
            "→",
            "↑",
        ]

    if env_name == "CliffWalkingEnv":

        return [
            "↑",
            "→",
            "↓",
            "←",
        ]

    raise NotImplementedError(
        f"{env_name} is not a supported grid environment."
    )


def get_action_vectors(env) -> dict[int, tuple[float, float]] | None:

    env = _unwrap_env(env)

    env_name = type(env).__name__

    if env_name == "FrozenLakeEnv":

        return {
            0: (-1.0, 0.0),   # ←
            1: (0.0, 1.0),    # ↓
            2: (1.0, 0.0),    # →
            3: (0.0, -1.0),   # ↑
        }

    if env_name == "CliffWalkingEnv":

        return {
            0: (0.0, -1.0),   # ↑
            1: (1.0, 0.0),    # →
            2: (0.0, 1.0),    # ↓
            3: (-1.0, 0.0),   # ←
        }

    raise NotImplementedError(
        f"{env_name} is not a supported grid environment."
    )

def compute_taxi_env_spatial_metrics(env, v_function, q_table, state_count, final_policy):
    """
    """
    # Initialize 5x5 matrices representing the physical Taxi map coordinates
    v_spatial = np.zeros((5, 5), dtype=np.float32)
    q_spatial = np.zeros((5, 5), dtype=np.float32)
    visits_spatial = np.zeros((5, 5), dtype=np.float32)
    policy_spatial = np.zeros((5, 5), dtype=np.float32)
    state_counts = np.zeros((5, 5), dtype=np.float32)

    # Iterate over all 500 environment states to decode and aggregate them spatially
    for state in range(env.observation_space.n):
        # unwrap and decode extract: (taxi_row, taxi_col, passenger_location, destination)
        taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
        
        v_spatial[taxi_row, taxi_col] += v_function[state]
        q_spatial[taxi_row, taxi_col] += np.mean(q_table[state]) # Average Q-values of actions at this grid position
        visits_spatial[taxi_row, taxi_col] += state_count[state]
        policy_spatial[taxi_row, taxi_col] += final_policy[state]
        state_counts[taxi_row, taxi_col] += 1.0

    # Compute the average value per grid cell to eliminate passenger/destination combination bias
    v_spatial /= state_counts
    q_spatial /= state_counts
    # For the policy mapping, round to the most predominant action chosen in that physical cell
    policy_spatial = np.round(policy_spatial / state_counts).astype(np.int32)

    return v_spatial, q_spatial, visits_spatial, policy_spatial