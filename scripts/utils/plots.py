import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colormaps

from typing import List, Any
from matplotlib.axes import Axes

from utils.utils import get_terminal_states_mask

def plot_avg_cumulative_reward(ax: Axes, running_average_rewards: List[float], env: Any, max_possible_reward: float) -> None:
    """
    Plots the average cumulative reward over time on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        running_average_rewards (list): A list of average cumulative rewards at each step or episode.
        env (gym.Env): The environment instance.
        max_possible_reward (float): The maximum possible reward in the environment.
    """
    ax.plot(running_average_rewards, color="tab:green", linewidth=2, label="Agent Performance")
    ax.axhline(y=max_possible_reward, color="tab:red", linestyle="--", label="Theoretical Optimum")
    ax.set_title("Average Cumulative Reward Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Running Average Reward")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

def plot_estimation_error(ax: Axes, mae_history: List[float], table_name: str) -> None:
    """
    Plots the estimation error (MAE) of the learned table (Q-Table or V-Table) over time on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        mae_history (list): A list of estimation errors at each step or episode.
        table_name (str): The name of the table for the label and title.
    """
    ax.plot(mae_history, color="tab:blue", linewidth=2)
    ax.set_title(f"{table_name} Estimation Error (MAE) Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Mean Absolute Error")
    ax.grid(True, linestyle=":", alpha=0.6)

def plot_decay_schedule(ax: Axes, schedule: np.ndarray, parameter_name: str = "Hyperparameter") -> None:
    """
    Plots the evolution of a decay schedule (e.g., epsilon or temperature) over time.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        schedule (numpy.ndarray): The precalculated array of decayed values.
        parameter_name (str): The name of the hyperparameter for the label and title (default: "Hyperparameter").
    """
    ax.plot(schedule, color="tab:orange", linewidth=2, label=f"Current {parameter_name}")
    ax.axhline(y=schedule[-1], color="black", linestyle=":", alpha=0.7, label=f"Min Value ({schedule[-1]:.3f})")
    ax.set_title(f"{parameter_name} Decay Schedule Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel(f"{parameter_name} Value")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

def plot_value_function_heatmap(V: np.ndarray, ax: plt.Axes):
    """
    Plots the reshaped V* state-value function as a heatmap on the given axis.
    
    Args:
        V (numpy.ndarray): The optimal value function V* for each state.
        ax (matplotlib.axes.Axes): The axes object to plot on.
    """
    mask, side = get_terminal_states_mask(len(V))
    V_grid = V.reshape(side, side)
    
    sns.heatmap(V_grid, annot=False, cmap=colormaps["YlGnBu"], cbar=True, ax=ax,
                square=True, cbar_kws={'label': 'Value'})

    for r in range(side):
        for c in range(side):
            if mask[r, c]:
                ax.text(c + 0.5, r + 0.5, f"{V_grid[r, c]:.3f}", color="black",
                        ha="center", va="center", fontsize=9)
            else:
                label = "G" if (r == side - 1 and c == side - 1) else "H"
                ax.text(c + 0.5, r + 0.5, f"[{label}]", color="crimson", 
                        ha="center", va="center", fontweight="bold", fontsize=10)

    ax.set_title("Optimal Value Function V*", fontsize=12, fontweight='bold')

def plot_q_function_heatmap(Q: np.ndarray, ax: plt.Axes):
    """
    Plots the full Q*(s, a) matrix as a detailed state-action heatmap.
    
    Args:
        Q (numpy.ndarray): The Q-value matrix.
        ax (matplotlib.axes.Axes): The axes object to plot on.
    """
    sns.heatmap(Q, annot=True, fmt=".3f", cmap=colormaps["magma"], cbar=True, ax=ax,
                xticklabels=["Left (0)", "Down (1)", "Right (2)", "Up (3)"],
                yticklabels=[f"S {i}" for i in range(len(Q))])
    
    ax.set_title("Optimal Action-Value Function Q*(s, a)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Actions")
    ax.set_ylabel("States")

def plot_optimal_policy_quiver(V: np.ndarray, pi: np.ndarray, ax: plt.Axes):
    """
    Plots decision vectors (arrows) of pi* mapped over the V* background.
    
    Args:
        V (numpy.ndarray): The optimal value function V* for each state.
        pi (numpy.ndarray): The optimal policy pi* for each state.
        ax (matplotlib.axes.Axes): The axes object to plot on.
    """
    mask, side = get_terminal_states_mask(len(V))
    V_grid = V.reshape(side, side)
    pi_grid = pi.reshape(side, side)
    
    # Direction mappings for actions
    # Azioni originali: 0=Left, 1=Down, 2=Right, 3=Up
    dx_map = {0: -1.0, 1: 0.0, 2: 1.0, 3: 0.0}
    dy_map = {0: 0.0, 1: 1.0, 2: 0.0, 3: -1.0}  # Inverted for image coordinates (Y increases downward)
    
    # Draw heatmap background
    sns.heatmap(V_grid, annot=False, cmap=colormaps["Blues"], cbar=False, ax=ax, square=True)
    
    # Plot arrows for each state
    for r in range(side):
        for c in range(side):
            if mask[r, c]:  # Only plot for non-terminal states
                action = int(pi_grid[r, c])
                dx = dx_map[action]
                dy = dy_map[action]
                x, y = c + 0.5, r + 0.5
                ax.arrow(x, y, dx * 0.35, dy * 0.35, head_width=0.18, head_length=0.12, 
                        fc='darkred', ec='darkred', linewidth=2)
    
    # Add labels for terminal states
    for r in range(side):
        for c in range(side):
            if not mask[r, c]:
                label = "G" if (r == side - 1 and c == side - 1) else "H"
                ax.text(c + 0.5, r + 0.5, f"[{label}]", color='crimson', 
                        ha='center', va='center', fontweight='bold', fontsize=10)

    ax.set_title("Optimal Policy pi* Quiver Map (Trajectories)", fontsize=12, fontweight='bold')