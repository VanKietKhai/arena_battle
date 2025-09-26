"""
Base AI Model Interface - Abstract contract for all AI implementations
All custom models must inherit from this class and implement required methods.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseAIModel(ABC):
    """
    Abstract base class for AI models in Arena Battle Game
    
    This interface ensures all AI models work seamlessly with the game infrastructure
    while allowing complete freedom in implementation details.
    """
    
    def __init__(self, model_name: str = "BaseModel", **kwargs):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training statistics - all models should track these
        self.episode_count = 0
        self.total_reward = 0.0
        self.kills = 0
        self.deaths = 0
        self.shots_fired = 0
        self.shots_hit = 0
        
        # Initialize model-specific components
        self._initialize_model(**kwargs)
    
    @abstractmethod
    def _initialize_model(self, **kwargs):
        """
        Initialize model-specific components (networks, optimizers, etc.)
        Called during __init__, implement your setup logic here.
        """
        pass
    
    @abstractmethod
    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate action from observation
        
        Args:
            observation: Processed game observation tensor
            deterministic: If True, use deterministic policy (no exploration)
        
        Returns:
            Tuple of (movement_action, aim_action, fire_action, log_prob)
            - movement_action: 2D tensor with x,y thrust values [-1, 1]
            - aim_action: 1D tensor with aim angle [0, 2π]
            - fire_action: 1D tensor with fire decision [0 or 1] 
            - log_prob: 1D tensor with action log probability
        """
        pass
    
    @abstractmethod
    def learn_from_experience(self, experience_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Update model from game experience
        
        Args:
            experience_data: Dictionary containing:
                - 'observation': Current observation
                - 'action': Action taken
                - 'reward': Reward received
                - 'next_observation': Next observation
                - 'done': Episode termination flag
                - 'additional_info': Any extra info (kills, deaths, etc.)
        
        Returns:
            Dictionary with training metrics (loss values, etc.)
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """
        Save complete model state
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """
        Load complete model state
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # Optional methods with default implementations
    def on_episode_start(self):
        """Called when a new episode starts"""
        pass
    
    def on_episode_end(self, episode_reward: float):
        """Called when episode ends"""
        self.episode_count += 1
        self.total_reward += episode_reward
    
    def on_kill(self):
        """Called when bot kills an enemy"""
        self.kills += 1
        self.shots_hit += 1  # Assume last shot was the killing shot
    
    def on_death(self):
        """Called when bot dies"""
        self.deaths += 1
    
    def on_shot_fired(self):
        """Called when bot fires a shot"""
        self.shots_fired += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current model statistics"""
        return {
            'model_name': self.model_name,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'kills': self.kills,
            'deaths': self.deaths,
            'kd_ratio': self.kills / max(self.deaths, 1),
            'shots_fired': self.shots_fired,
            'shots_hit': self.shots_hit,
            'accuracy': self.shots_hit / max(self.shots_fired, 1) * 100,
            'avg_reward': self.total_reward / max(self.episode_count, 1)
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.episode_count = 0
        self.total_reward = 0.0
        self.kills = 0
        self.deaths = 0
        self.shots_fired = 0
        self.shots_hit = 0
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information for display/logging"""
        return {
            'name': self.model_name,
            'type': self.__class__.__name__,
            'device': str(self.device),
            'parameters': str(self.count_parameters()) if hasattr(self, 'count_parameters') else 'N/A'
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters (override for custom implementations)"""
        return 0


class ObservationProcessor:
    """
    Standardized observation processor for all AI models
    Converts raw game observations into normalized tensor format
    """
    
    def __init__(self, obs_dim: int = 48):
        self.obs_dim = obs_dim
    
    def process(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Convert observation dict to standardized tensor format
        
        This is the same processing logic from the original network.py
        All models receive observations in this standardized format
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Arena dimensions
        arena_width = obs_dict.get('arena_width', 800)
        arena_height = obs_dict.get('arena_height', 600)
        
        # Self state (normalized)
        self_pos = obs_dict.get('self_pos', {'x': 0, 'y': 0})
        obs[0] = self_pos['x'] / arena_width
        obs[1] = self_pos['y'] / arena_height
        obs[2] = obs_dict.get('self_hp', 100) / 100.0
        
        # Enemy state
        enemy_pos = obs_dict.get('enemy_pos', {'x': 0, 'y': 0})
        obs[3] = enemy_pos['x'] / arena_width
        obs[4] = enemy_pos['y'] / arena_height
        obs[5] = obs_dict.get('enemy_hp', 0) / 100.0
        
        # Distance and angle to enemy
        dx = enemy_pos['x'] - self_pos['x']
        dy = enemy_pos['y'] - self_pos['y']
        distance = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        
        obs[6] = distance / 1000.0
        obs[7] = angle / np.pi  # Normalized to [-1, 1]
        
        # Bullet info
        bullets = obs_dict.get('bullets', [])
        obs[8] = min(len(bullets), 10) / 10.0
        
        # Line of sight (IMPORTANT for smart firing)
        obs[9] = float(obs_dict.get('has_line_of_sight', False))
        
        # Arena bounds
        obs[10] = arena_width / 1000.0
        obs[11] = arena_height / 1000.0
        
        # Combat state features
        obs[12] = 1.0 if distance < 200 else 0.0  # Close to enemy
        obs[13] = 1.0 if len(bullets) > 0 else 0.0  # Bullets nearby
        obs[14] = 1.0 if obs_dict.get('has_line_of_sight', False) and distance < 300 else 0.0  # Good shot opportunity
        
        # === WALL AVOIDANCE FEATURES ===
        
        # Calculate distances to arena boundaries
        left_dist = self_pos['x'] / arena_width  # Distance to left wall (0-1)
        right_dist = (arena_width - self_pos['x']) / arena_width  # Distance to right wall
        top_dist = self_pos['y'] / arena_height  # Distance to top wall
        bottom_dist = (arena_height - self_pos['y']) / arena_height  # Distance to bottom wall
        
        # Wall distances (normalized, 0 = at wall, 1 = far from wall)
        obs[15] = left_dist
        obs[16] = right_dist
        obs[17] = top_dist
        obs[18] = bottom_dist
        
        # Wall proximity warnings (1 if too close to wall)
        wall_warning_threshold = 0.1  # 10% of arena size
        obs[19] = 1.0 if left_dist < wall_warning_threshold else 0.0
        obs[20] = 1.0 if right_dist < wall_warning_threshold else 0.0
        obs[21] = 1.0 if top_dist < wall_warning_threshold else 0.0
        obs[22] = 1.0 if bottom_dist < wall_warning_threshold else 0.0
        
        # Wall avoidance directions (for movement bias)
        obs[23] = 1.0 if left_dist < 0.2 else 0.0    # Should move right
        obs[24] = -1.0 if right_dist < 0.2 else 0.0  # Should move left
        obs[25] = 1.0 if top_dist < 0.2 else 0.0     # Should move down
        obs[26] = -1.0 if bottom_dist < 0.2 else 0.0 # Should move up
        
        # === SMART AIMING FEATURES ===
        
        # Angle difference between current aim and enemy direction
        obs[27] = np.cos(angle)  # X component of enemy direction
        obs[28] = np.sin(angle)  # Y component of enemy direction
        
        # Enemy movement prediction (simple)
        obs[29] = dx / arena_width   # Enemy relative X position
        obs[30] = dy / arena_height  # Enemy relative Y position
        
        # === TACTICAL FEATURES ===
        
        # Corner positions (good for defensive play)
        corners = [
            (50, 50), (arena_width-50, 50), 
            (50, arena_height-50), (arena_width-50, arena_height-50)
        ]
        
        min_corner_dist = float('inf')
        for corner_x, corner_y in corners:
            corner_dist = np.sqrt((self_pos['x'] - corner_x)**2 + (self_pos['y'] - corner_y)**2)
            min_corner_dist = min(min_corner_dist, corner_dist)
        
        obs[31] = min_corner_dist / 200.0  # Distance to nearest corner
        
        # Center control (good for aggressive play)
        center_x, center_y = arena_width / 2, arena_height / 2
        center_dist = np.sqrt((self_pos['x'] - center_x)**2 + (self_pos['y'] - center_y)**2)
        obs[32] = center_dist / 300.0  # Distance to center
        
        # === BULLET THREAT ANALYSIS ===
        
        # Analyze nearby bullets for threat level
        bullet_threat = 0.0
        for bullet in bullets:
            bullet_dx = bullet['x'] - self_pos['x']
            bullet_dy = bullet['y'] - self_pos['y']
            bullet_dist = np.sqrt(bullet_dx*bullet_dx + bullet_dy*bullet_dy)
            
            if bullet_dist < 100:  # Nearby bullet
                bullet_threat += (100 - bullet_dist) / 100.0
        
        obs[33] = min(bullet_threat, 1.0)  # Bullet threat level
        
        # === FIRING OPPORTUNITY ASSESSMENT ===
        
        # Good shot conditions
        good_shot = (
            obs_dict.get('has_line_of_sight', False) and  # Can see enemy
            distance < 400 and  # Enemy in range
            distance > 50   # Not too close (avoid friendly fire area)
        )
        obs[34] = 1.0 if good_shot else 0.0
        
        # Enemy visibility duration (would need tracking, using LOS as proxy)
        obs[35] = float(obs_dict.get('has_line_of_sight', False))
        
        # === REMAINING FEATURES (padding) ===
        
        # Fill remaining slots with useful derived features
        obs[36] = np.sin(2 * angle)  # Harmonic of enemy angle
        obs[37] = np.cos(2 * angle)  # Harmonic of enemy angle
        obs[38] = 1.0 if distance < 150 else 0.0  # Very close combat
        obs[39] = 1.0 if distance > 500 else 0.0  # Long range combat
        
        # Health ratio features
        enemy_hp = obs_dict.get('enemy_hp', 0)
        self_hp = obs_dict.get('self_hp', 100)
        health_advantage = (self_hp - enemy_hp) / 100.0
        obs[40] = health_advantage
        obs[41] = 1.0 if health_advantage > 0 else 0.0  # Winning
        obs[42] = 1.0 if health_advantage < -0.5 else 0.0  # Critical health disadvantage
        
        # Movement encouragement (anti-camping)
        obs[43] = 1.0  # Always encourage movement
        obs[44] = np.random.uniform(0, 1)  # Random exploration signal
        
        # Arena position category
        edge_threshold = 0.15
        is_near_edge = (left_dist < edge_threshold or right_dist < edge_threshold or 
                       top_dist < edge_threshold or bottom_dist < edge_threshold)
        obs[45] = 1.0 if is_near_edge else 0.0
        
        # Final tactical signals
        obs[46] = 1.0 if good_shot and health_advantage > 0 else 0.0  # Attack opportunity
        obs[47] = 1.0 if bullet_threat > 0.5 or health_advantage < -0.3 else 0.0  # Retreat signal
        
        return torch.FloatTensor(obs).unsqueeze(0)