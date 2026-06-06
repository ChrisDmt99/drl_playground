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