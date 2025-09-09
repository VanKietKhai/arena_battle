"""
AI Models Package - Modular AI Architecture
This package contains all AI model implementations for the Arena Battle Game.

Available Models:
- PPOModel: Enhanced PPO with wall avoidance and smart aiming
- (Future: DQNModel, A3CModel, RandomModel, etc.)

Usage:
    from ai_bot.core.model_loader import create_model
    model = create_model('ppo')  # Creates PPO model
"""

from .base_model import BaseAIModel, ObservationProcessor
from .ppo_model import PPOModel

# Export main classes
__all__ = [
    'BaseAIModel',
    'ObservationProcessor', 
    'PPOModel'
]

# Model registry for easy access
AVAILABLE_MODELS = {
    'ppo': PPOModel,
}

def get_available_model_types():
    """Get list of available model types"""
    return list(AVAILABLE_MODELS.keys())

def create_model_instance(model_type: str, **kwargs):
    """Create a model instance by type"""
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = AVAILABLE_MODELS[model_type]
    return model_class(**kwargs)