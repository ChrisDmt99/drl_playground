import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors

from typing import List, Any
from matplotlib.axes import Axes

# from utils.utils import get_terminal_states_mask

def plot_avg_cumulative_reward(ax: Axes, running_average_rewards: List[float], title:str, env: Any, theoretical_return: float, asymptote_label: str) -> None:
    """
    Plots the average cumulative reward over time on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        running_average_rewards (list): A list of average cumulative rewards at each step or episode.
        title (str): The title for the plot.
        env (gym.Env): The environment instance.
        theoretical_return (float): The theoretical expected value of returns.
        asymptote_label (str): The label for the asymptotic line.
    """
    ax.plot(running_average_rewards, color="tab:green", linewidth=2, label="Agent Performance")
    ax.axhline(y=theoretical_return, color="tab:red", linestyle="--", label=f"{asymptote_label} ({theoretical_return:.3f})")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel(title)
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
    ax.set_title(f"{table_name} Estimation Error (MAE)", fontsize=12, fontweight="bold")
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
    # ax.axhline(y=schedule[-1], color="black", linestyle=":", alpha=0.7, label=f"Min Value ({schedule[-1]:.3f})")
    ax.set_title(f"{parameter_name} Decay Schedule", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episodes")
    ax.set_ylabel(f"{parameter_name}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

def plot_total_regret(ax: plt.Axes, total_regret_history: list):
    """
    Plots the cumulative total regret over political selection episodes.
    An optimal policy should show a sub-linear growth (logarithmic).
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        total_regret_history (list): Array containing the historical total sum of regret.
    """
    ax.plot(total_regret_history, color="crimson", linewidth=2, label="Total Regret")
    
    ax.set_title("Total Cumulative Regret", fontsize=12, fontweight='bold')
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Accumulated Regret")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

def plot_q_function_heatmap(
    Q: np.ndarray,
    ax: plt.Axes,
    action_names: list[str] | None = None,
    terminal_states: list[int] | None = None,
    goal_states: list[int] | None = None,
    special_states: list[int] | None = None,
):
    n_states, n_actions = Q.shape
    terminal_states = set(terminal_states or [])
    goal_states = set(goal_states or [])
    special_states = set(special_states or [])

    if action_names is None:
        action_names = ["↑", "→", "↓", "←"]

    # Esclusione stati speciali
    all_special = terminal_states | goal_states | special_states
    mask = np.ones(Q.shape, dtype=bool)
    for s in all_special:
        mask[s, :] = False
    
    # FILTRO: Escludiamo ostacoli (<= -100) e azioni nulle (valori vicini allo zero)
    # np.abs(Q) > 1e-3 impedisce agli zeri di schiacciare la scala
    valid_mask = mask & (Q > -100) & (np.abs(Q) > 1e-3)
    valid_Q = Q[valid_mask]
    
    vmin = np.min(valid_Q) if valid_Q.size > 0 else 0
    vmax = np.max(Q) if Q.size > 0 else 1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu").copy()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    
    grid_colors = cmap(norm(Q))
    
    # Impostiamo colore neutro (grigio chiaro) per le azioni con valore ~0
    zero_mask = (np.abs(Q) <= 1e-3)
    grid_colors[zero_mask] = [0.95, 0.95, 0.95, 1.0]

    # Sovrascriviamo gli stati speciali
    colors = {
        'T': mcolors.to_rgba("#d9d9d9"),
        'S': mcolors.to_rgba("#8d6e63"),
        'G': mcolors.to_rgba("#4caf50")
    }

    for s in range(n_states):
        if s in terminal_states: grid_colors[s, :] = colors['T']
        elif s in goal_states: grid_colors[s, :] = colors['G']
        elif s in special_states: grid_colors[s, :] = colors['S']

    ax.imshow(grid_colors, aspect='auto')

    for r in range(n_states):
        for c in range(n_actions):
            if r in goal_states: text = "[G]"
            elif r in special_states: text = "[S]"
            elif r in terminal_states: text = "[T]"
            else: text = f"{Q[r, c]:.2f}"
            ax.text(c, r, text, ha="center", va="center", color="black", fontsize=8, fontweight="bold")

    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks(np.arange(n_actions))
    ax.set_xticklabels(action_names)
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels([f"S{i}" for i in range(n_states)])
    ax.set_title("Action-Value Function Q(s,a)", fontweight="bold")
    ax.set_xlabel("Actions")
    ax.set_ylabel("States")
    ax.grid(False)

def plot_value_function_heatmap(
    V: np.ndarray,
    rows: int,
    cols: int,
    ax: plt.Axes,
    terminal_states: list[int] | None = None,
    goal_states: list[int] | None = None,
    special_states: list[int] | None = None,
):
    terminal_states = set(terminal_states or [])
    goal_states = set(goal_states or [])
    special_states = set(special_states or [])

    #### Esclude terminal_states, goal_states and special_states
    V_grid = V.reshape(rows, cols)
    
    # Creiamo una maschera per escludere gli stati speciali dal calcolo di vmin/vmax
    all_special = terminal_states | goal_states | special_states
    mask = np.ones(V.shape, dtype=bool)
    for s in all_special:
        mask[s] = False
    
    # Filtriamo i valori validi: non speciali e non ostacoli (>-100)
    valid_mask = mask & (V > -100)
    valid_V = V[valid_mask]
    
    vmin = np.min(valid_V) if valid_V.size > 0 else 0
    vmax = np.max(valid_V) if valid_V.size > 0 else 1
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu").copy()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    grid_colors = cmap(norm(V_grid))
    colors = {
        'T': mcolors.to_rgba("#d9d9d9"),
        'S': mcolors.to_rgba("#8d6e63"),
        'G': mcolors.to_rgba("#4caf50")
    }

    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            if s in terminal_states: grid_colors[r, c] = colors['T']
            elif s in goal_states: grid_colors[r, c] = colors['G']
            elif s in special_states: grid_colors[r, c] = colors['S']

    ax.imshow(grid_colors)
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            if s in goal_states: text = "[G]"
            elif s in special_states: text = "[S]"
            elif s in terminal_states: text = "[T]"
            else: text = f"{V_grid[r, c]:.2f}"
            ax.text(c, r, text, ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_title("State-Value Function V(s)", fontweight="bold")
    ax.grid(False)

def plot_policy_quiver(
    pi: np.ndarray, 
    rows: int,
    cols: int,
    ax: plt.Axes,
    action_vectors: dict[int, tuple[float, float]] | None = None,
    terminal_states: list[int] | None = None,
    goal_states: list[int] | None = None,
    special_states: dict[int, str] | None = None,
):
    """
    Generic GridWorld policy visualization (uniform color for normal states).
    """
    terminal_states = set(terminal_states or [])
    goal_states = set(goal_states or [])
    special_states = special_states or {}

    if action_vectors is None:
        action_vectors = {0: (-1.0, 0.0), 1: (0.0, 1.0), 2: (1.0, 0.0), 3: (0.0, -1.0)}

    # Griglia base: 0.0 per celle normali
    grid = np.zeros((rows, cols))
    
    # Valori sentinella fuori dal range [-0.5, 0.5]
    T_VAL, G_VAL = -1.0, 1.0
    
    for s in range(rows * cols):
        r, c = divmod(s, cols)
        if s in terminal_states: 
            grid[r, c] = T_VAL
        elif s in goal_states: 
            grid[r, c] = G_VAL
        elif s in special_states: 
            grid[r, c] = np.nan  

    # 3. Configurazione Colormap
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("#d9d9d9") 
    cmap.set_over("#4caf50")  
    cmap.set_bad("#8d6e63")   
    
    sns.heatmap(
        grid,
        cmap=cmap,
        vmin=-0.5, 
        vmax=0.5,
        annot=False,
        square=True,
        cbar=False,
        ax=ax
    )

    # Disegno testi e frecce
    for s in range(rows * cols):
        r, c = divmod(s, cols)
        x, y = c + 0.5, r + 0.5

        if s in goal_states:
            ax.text(x, y, "[G]", ha="center", va="center", fontweight="bold", color="black")
        elif s in special_states:
            ax.text(x, y, "[S]", ha="center", va="center", fontweight="bold", color="black")
        elif s in terminal_states:
            ax.text(x, y, "[T]", ha="center", va="center", fontweight="bold", color="black")
        else:
            action = int(pi[s])
            if action in action_vectors:
                dx, dy = action_vectors[action]
                ax.arrow(x, y, dx * 0.35, dy * 0.35, head_width=0.18, head_length=0.12, 
                         fc="darkred", ec="darkred", linewidth=2, length_includes_head=True)

    ax.set_title("Policy Quiver Map", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(np.arange(rows))
    
    # Rimuovi i bordi esterni ma mantieni i tick
    for spine in ax.spines.values(): 
        spine.set_visible(False)
        
    ax.tick_params(left=True, bottom=True)