"""
Dynamic Model Loading System
Automatically discovers and loads AI models from the models directory
"""

import importlib
import os
import inspect
import logging
from pathlib import Path
from typing import Dict, Type, Optional, Any

from ..models.base_model import BaseAIModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for available AI model implementations"""
    
    def __init__(self):
        self._models: Dict[str, Type[BaseAIModel]] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._discover_models()
    
    def _discover_models(self):
        """Automatically discover and register models from models directory"""
        models_dir = Path(__file__).parent.parent / "models"
        
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return
        
        # Scan all Python files in models directory
        for model_file in models_dir.glob("*.py"):
            if model_file.name.startswith("_") or model_file.name == "base_model.py":
                continue  # Skip private files and base class
            
            try:
                # Import module dynamically
                module_name = f"ai_bot.models.{model_file.stem}"
                module = importlib.import_module(module_name)
                
                # Find BaseAIModel subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseAIModel) and 
                        obj != BaseAIModel and
                        obj.__module__ == module.__name__):
                        
                        model_key = self._generate_model_key(name, model_file.stem)
                        self._register_model(model_key, obj, module_name)
                        
                        logger.info(f"📦 Discovered model: {model_key} ({name})")
                        
            except Exception as e:
                logger.error(f"❌ Failed to load model from {model_file}: {e}")
        
        logger.info(f"✅ Model discovery complete: {len(self._models)} models available")
    
    def _generate_model_key(self, class_name: str, file_name: str) -> str:
        """Generate a user-friendly model key"""
        # Try to extract a clean name
        if file_name.endswith("_model"):
            return file_name[:-6]  # Remove "_model" suffix
        elif class_name.endswith("Model"):
            return class_name[:-5].lower()  # Remove "Model" suffix and lowercase
        else:
            return file_name
    
    def _register_model(self, key: str, model_class: Type[BaseAIModel], module_name: str):
        """Register a model class"""
        self._models[key] = model_class
        self._model_info[key] = {
            'class_name': model_class.__name__,
            'module': module_name,
            'docstring': inspect.getdoc(model_class) or "No description available"
        }
    
    def get_model_class(self, model_key: str) -> Optional[Type[BaseAIModel]]:
        """Get model class by key"""
        return self._models.get(model_key)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their information"""
        return self._model_info.copy()
    
    def get_model_keys(self) -> list:
        """Get list of all model keys"""
        return list(self._models.keys())
    
    def model_exists(self, model_key: str) -> bool:
        """Check if a model exists"""
        return model_key in self._models


class ModelLoader:
    """Factory class for creating AI model instances"""
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def create_model(self, model_type: str, **kwargs) -> Optional[BaseAIModel]:
        """
        Create an AI model instance
        
        Args:
            model_type: Type of model to create (e.g., 'ppo', 'dqn')
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            BaseAIModel instance or None if creation failed
        """
        model_class = self.registry.get_model_class(model_type)
        
        if model_class is None:
            logger.error(f"❌ Unknown model type: {model_type}")
            self.list_available_models()
            return None
        
        try:
            logger.info(f"🔧 Creating {model_type} model...")
            model = model_class(**kwargs)
            logger.info(f"✅ Successfully created {model_type} model")
            logger.info(f"📊 Model info: {model.get_model_info()}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create {model_type} model: {e}")
            return None
    
    def list_available_models(self):
        """List all available models for user reference"""
        models = self.registry.list_models()
        
        if not models:
            logger.warning("⚠️  No models available!")
            return
        
        logger.info("📋 Available AI Models:")
        for key, info in models.items():
            logger.info(f"  • {key} ({info['class_name']})")
            if info['docstring']:
                # Show first line of docstring
                first_line = info['docstring'].split('\n')[0]
                logger.info(f"    └─ {first_line}")
    
    def get_default_model_type(self) -> str:
        """Get the default model type (PPO if available)"""
        if self.registry.model_exists('ppo'):
            return 'ppo'
        
        # Return first available model
        keys = self.registry.get_model_keys()
        return keys[0] if keys else None
    
    def validate_model_type(self, model_type: str) -> bool:
        """Validate if model type is available"""
        return self.registry.model_exists(model_type)


# Global model loader instance
_model_loader = None

def get_model_loader() -> ModelLoader:
    """Get global model loader instance (singleton)"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def create_model(model_type: str, **kwargs) -> Optional[BaseAIModel]:
    """Convenience function to create a model"""
    loader = get_model_loader()
    return loader.create_model(model_type, **kwargs)


def list_available_models():
    """Convenience function to list available models"""
    loader = get_model_loader()
    loader.list_available_models()


def get_default_model_type() -> str:
    """Get the default model type"""
    loader = get_model_loader()
    return loader.get_default_model_type()