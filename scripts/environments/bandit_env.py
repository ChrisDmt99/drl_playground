import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
from typing import Optional, Tuple, Dict, Any

class MultiArmedBandit(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, num_arms: int, seed: int, render_mode: Optional[str] = None) -> None:
        """
        Initialize the Multi-Armed Bandit environment.

        Args:
            num_arms (int): The number of arms (actions) in the bandit.
            seed (int): The random seed for reproducibility.
            render_mode (str, optional): The rendering mode. Defaults to None.
        """
        super().__init__()
        self.num_arms = num_arms
        self.seed = seed
        self.local_rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        
        # Definition of the real probabilities
        self.true_probabilities = self.local_rng.uniform(0.0, 1.0, size=self.num_arms)
        
        # Debug
        print(f"[Bandit Env] Stable probabilities generated for seed {seed}: {self.true_probabilities}")
        
        # Action space
        self.action_space = spaces.Discrete(self.num_arms)

        # Observation space: there is only 1 constant state (state 0)
        self.observation_space = spaces.Discrete(1)
        
        # Rendering attributes
        self.window = None
        self.clock = None
        self.window_size = (600, 400)
        self.last_action = None
        self.last_reward = None       

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        
        Args:
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            Tuple[int, Dict[str, Any]]: The initial observation and an info dictionary containing the true probabilities of the arms.
        """
        super().reset(seed=self.seed)

        self.last_action = None
        self.last_reward = None
        
        if self.render_mode == "human":
            self.render()
            
        return 0, {"true_probabilities": self.true_probabilities}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        # Compute the reward based on the true probabilities of the chosen arm
        reward = 1.0 if np.random.rand() < self.true_probabilities[action] else 0.0
        
        self.last_action = action
        self.last_reward = reward
        
        # A bandit never finishes theoretically, but we send done=True 
        # if we want to simulate a "single-step episode"
        terminated = True 
        truncated = False
        
        if self.render_mode == "human":
            self.render()
            time.sleep(0.1) # A small delay per render the animation visible to the human eye

        return 0, reward, terminated, truncated, {}

    def render(self) -> None:
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Multi-Armed Bandit Casino")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((20, 20, 30))  # Dark Casino Background
        
        # Dynamically calculate button dimensions based on number of arms
        available_width = self.window_size[0] - 60  # 30px margin on each side
        min_spacing = 20
        total_spacing = min_spacing * (self.num_arms - 1) if self.num_arms > 1 else 0
        
        # Calculate button width to fit all arms
        button_width = max(60, (available_width - total_spacing) // self.num_arms)
        button_height = 140
        spacing = min_spacing
        
        # Center the buttons
        total_width = button_width * self.num_arms + spacing * (self.num_arms - 1)
        start_x = (self.window_size[0] - total_width) // 2
        y_pos = 100

        # Responsive font sizes
        title_font_size = max(20, min(28, 28 - (self.num_arms - 4)))
        button_font_size = max(12, min(18, 18 - (self.num_arms - 4) // 2))
        status_font_size = max(12, 16 - (self.num_arms - 4) // 3)

        font = pygame.font.SysFont("Arial", button_font_size)
        title_font = pygame.font.SysFont("Arial", title_font_size, bold=True)
        status_font = pygame.font.SysFont("Arial", status_font_size)

        # Title with gradient effect (using layered text)
        title_text = title_font.render("Multi-Armed Bandit", True, (255, 215, 0))
        title_shadow = title_font.render("Multi-Armed Bandit", True, (100, 80, 0))
        canvas.blit(title_shadow, (self.window_size[0] // 2 - title_text.get_width() // 2 + 2, 28))
        canvas.blit(title_text, (self.window_size[0] // 2 - title_text.get_width() // 2, 25))

        # Draw arms/buttons with improved styling
        for i in range(self.num_arms):
            x_pos = start_x + i * (button_width + spacing)
            
            # Default color: subtle blue-grey
            color = (60, 70, 90)
            border_color = (150, 160, 180)
            text_color = (220, 230, 240)
            
            # If this was the last action chosen, color it according to the reward
            if self.last_action == i:
                if self.last_reward == 1.0:
                    color = (46, 204, 113)  # Green for victory
                    border_color = (76, 224, 143)
                    text_color = (255, 255, 255)
                else:
                    color = (231, 76, 60)   # Red for defeat
                    border_color = (255, 106, 90)
                    text_color = (255, 255, 255)
            
            # Draw shadow effect
            pygame.draw.rect(canvas, (0, 0, 0), (x_pos + 2, y_pos + 2, button_width, button_height), 
                           border_radius=6)
            
            # Draw main button
            pygame.draw.rect(canvas, color, (x_pos, y_pos, button_width, button_height), border_radius=6)
            
            # Draw border
            pygame.draw.rect(canvas, border_color, (x_pos, y_pos, button_width, button_height), 
                           width=2, border_radius=6)
            
            # Draw probability indicator (subtle bar at the top of each button)
            prob = self.true_probabilities[i]
            prob_bar_width = int((button_width - 4) * prob)
            pygame.draw.rect(canvas, (100, 200, 100), (x_pos + 2, y_pos + 2, prob_bar_width, 8))
            
            # Text on the button
            txt = font.render(f"Arm {i}", True, text_color)
            txt_rect = txt.get_rect(center=(x_pos + button_width // 2, y_pos + button_height // 2 - 10))
            canvas.blit(txt, txt_rect)
            
            # Display probability (smaller text)
            prob_txt = pygame.font.SysFont("Arial", max(9, button_font_size - 4)).render(
                f"p={prob:.2f}", True, text_color)
            prob_txt_rect = prob_txt.get_rect(center=(x_pos + button_width // 2, y_pos + button_height - 15))
            canvas.blit(prob_txt, prob_txt_rect)

        # Informative text at the bottom
        if self.last_reward is not None:
            status_str = "✓ VICTORY (+1.0)" if self.last_reward == 1.0 else "✗ DEFEAT (0.0)"
            status_color = (76, 224, 143) if self.last_reward == 1.0 else (255, 106, 90)
            status_text = status_font.render(
                f"Last Action: Arm {self.last_action} → {status_str}", True, status_color)
            canvas.blit(status_text, (self.window_size[0] // 2 - status_text.get_width() // 2, 
                                     self.window_size[1] - 50))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()