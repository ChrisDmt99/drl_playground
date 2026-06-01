import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any
from matplotlib.axes import Axes

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