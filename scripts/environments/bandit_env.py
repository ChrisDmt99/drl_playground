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
        canvas.fill((30, 30, 30)) # Dark Casino Background 
        
        # Let's draw the 4 arms/buttons
        button_width = 100
        button_height = 150
        spacing = 40
        start_x = (self.window_size[0] - (button_width * self.num_arms + spacing * (self.num_arms - 1))) // 2
        y_pos = 120

        font = pygame.font.SysFont("Arial", 18)
        title_font = pygame.font.SysFont("Arial", 28, bold=True)

        # Higher title
        title_text = title_font.render("Multi-Armed Bandit", True, (255, 215, 0))
        canvas.blit(title_text, (self.window_size[0] // 2 - title_text.get_width() // 2, 30))

        for i in range(self.num_arms):
            x_pos = start_x + i * (button_width + spacing)
            
            # Grey Higher title
            color = (100, 100, 100)
            
            # If this was the last action chosen, color it according to the reward
            if self.last_action == i:
                if self.last_reward == 1.0:
                    color = (46, 204, 113) # Green for victory
                else:
                    color = (231, 76, 60)  # Red for defeat

            # Draw the slot button
            pygame.draw.rect(canvas, color, (x_pos, y_pos, button_width, button_height), border_radius=8)
            pygame.draw.rect(canvas, (255, 255, 255), (x_pos, y_pos, button_width, button_height), width=2, border_radius=8)
            
            # Text on the button
            txt = font.render(f"Arm {i}", True, (255, 255, 255))
            canvas.blit(txt, (x_pos + button_width//2 - txt.get_width()//2, y_pos + button_height//2 - 10))

        # Informative text at the bottom
        if self.last_reward is not None:
            status_str = "VICTORY (+1.0)" if self.last_reward == 1.0 else "DEFEAT (0.0)"
            status_color = (46, 204, 113) if self.last_reward == 1.0 else (231, 76, 60)
            status_text = font.render(f"Last Action: Arm {self.last_action} -> {status_str}", True, status_color)
            canvas.blit(status_text, (self.window_size[0] // 2 - status_text.get_width() // 2, 320))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()