"""
AI Bot Core Package - Infrastructure Components
Contains the core infrastructure that connects AI models to the game server.

This package handles:
- Model loading and registration
- Game server communication
- Protocol buffer conversion
- Enhanced action processing

Components:
- model_loader: Dynamic model discovery and instantiation
- game_interface: Game server communication and protocol handling
"""

from .model_loader import (
    ModelLoader, 
    ModelRegistry, 
    create_model, 
    list_available_models, 
    get_default_model_type
)
from .game_interface import GameInterface

__all__ = [
    'ModelLoader',
    'ModelRegistry', 
    'create_model',
    'list_available_models',
    'get_default_model_type',
    'GameInterface'
]