"""
AI Bot Main Entry Point - Refactored with Modular Architecture
Supports pluggable AI models through the models/ directory
"""

import asyncio
import logging
import argparse
import sys
import os
import signal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_bot.core.model_loader import create_model, list_available_models, get_default_model_type
from ai_bot.core.game_interface import GameInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global interface for graceful shutdown
game_interface = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully with auto-save"""
    global game_interface
    if game_interface and game_interface.model:
        logger.info("🛑 Received shutdown signal - saving model...")
        asyncio.create_task(game_interface._save_model_on_exit())
    sys.exit(0)

def find_latest_model(player_id, models_dir):
    """Find the latest model for a player"""
    models_path = Path(models_dir)
    if not models_path.exists():
        return None
    
    # Look for models matching player ID pattern
    pattern = f"{player_id}_*.pth"
    model_files = list(models_path.glob(pattern))
    
    if not model_files:
        return None
    
    # Sort by modification time, newest first
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(model_files[0])

def list_player_models(player_id, models_dir):
    """List available models for a player"""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    pattern = f"{player_id}_*.pth"
    model_files = list(models_path.glob(pattern))
    
    # Sort by modification time, newest first
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    models_info = []
    for model_file in model_files:
        try:
            import torch
            checkpoint = torch.load(model_file, map_location='cpu')
            info = {
                'file': str(model_file),
                'name': model_file.name,
                'save_type': checkpoint.get('save_type', 'unknown'),
                'model_type': checkpoint.get('statistics', {}).get('model_name', 'Unknown'),
                'kd_ratio': checkpoint.get('statistics', {}).get('kd_ratio', 0),
                'accuracy': checkpoint.get('statistics', {}).get('accuracy', 0),
                'episodes': checkpoint.get('statistics', {}).get('episode_count', 0),
                'save_time': checkpoint.get('save_time', 'unknown')
            }
            models_info.append(info)
        except Exception as e:
            logger.warning(f"⚠️ Could not read model {model_file}: {e}")
    
    return models_info

async def main():
    """Main entry point with modular AI model support"""
    global game_interface
    
    parser = argparse.ArgumentParser(description='Arena Battle AI Bot - Modular Architecture')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--player-id', required=True, help='Unique player ID')
    parser.add_argument('--bot-name', help='Bot name (default: enhanced player ID)')
    
    # Model selection
    parser.add_argument('--model-type', help='AI model type (ppo, dqn, etc.)')
    parser.add_argument('--list-models', action='store_true', help='List available AI model types')
    parser.add_argument('--list-saved-models', action='store_true', help='List saved models for player')
    
    # Model loading
    parser.add_argument('--model-path', help='Path to specific model file to load')
    parser.add_argument('--auto-load', action='store_true', help='Auto-load latest model for player')
    
    # Directories
    parser.add_argument('--models-dir', default='models/checkpoints', help='Models directory')
    
    args = parser.parse_args()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    if not args.bot_name:
        args.bot_name = f"AI_{args.player_id}"
    
    # List AI model types if requested
    if args.list_models:
        logger.info("🤖 Available AI Model Types:")
        list_available_models()
        return
    
    # List saved models if requested
    if args.list_saved_models:
        logger.info(f"💾 Saved models for player '{args.player_id}':")
        models = list_player_models(args.player_id, args.models_dir)
        
        if not models:
            logger.info("❌ No saved models found for this player")
        else:
            for i, model in enumerate(models):
                logger.info(f"  {i+1}. {model['name']}")
                logger.info(f"     Type: {model['model_type']}, K/D: {model['kd_ratio']:.2f}")
                logger.info(f"     Accuracy: {model['accuracy']:.1f}%, Episodes: {model['episodes']}")
                logger.info(f"     Saved: {model['save_time']}")
                logger.info("")
        return
    
    # Determine model type to use
    model_type = args.model_type
    if not model_type:
        model_type = get_default_model_type()
        if not model_type:
            logger.error("❌ No AI models available! Check models/ directory")
            return
        logger.info(f"🎯 Using default model type: {model_type}")
    
    # Display startup banner
    logger.info("🤖 ==========================================")
    logger.info("🤖   ARENA BATTLE AI BOT - MODULAR")
    logger.info("🤖 ==========================================")
    logger.info(f"🤖 Bot Name: {args.bot_name}")
    logger.info(f"🤖 Player ID: {args.player_id}")
    logger.info(f"🧠 AI Model: {model_type}")
    logger.info(f"🌐 Server: {args.host}:{args.port}")
    logger.info("⚔️ Mode: PvP Combat")
    logger.info(f"💾 Models Directory: {args.models_dir}")
    
    # Create AI model
    logger.info(f"🔧 Initializing {model_type} model...")
    model = create_model(model_type)
    
    if model is None:
        logger.error(f"❌ Failed to create {model_type} model")
        logger.info("💡 Available models:")
        list_available_models()
        return
    
    # Handle model loading
    model_loaded = False
    model_to_load = None
    
    if args.model_path:
        # Specific model requested
        if os.path.exists(args.model_path):
            model_to_load = args.model_path
            logger.info(f"🎯 Loading specific model: {args.model_path}")
        else:
            logger.error(f"❌ Model file not found: {args.model_path}")
            return
    elif args.auto_load:
        # Auto-load latest model
        model_to_load = find_latest_model(args.player_id, args.models_dir)
        if model_to_load:
            logger.info(f"📄 Auto-loading latest model: {model_to_load}")
        else:
            logger.info("🆕 No existing models found - starting fresh")
    else:
        # Check if models exist and offer to load
        latest_model = find_latest_model(args.player_id, args.models_dir)
        if latest_model:
            logger.info(f"💡 Found existing model: {Path(latest_model).name}")
            logger.info("   Use --auto-load to load it automatically")
            logger.info("   Use --list-saved-models to see all available models")
        logger.info("🆕 Starting with fresh neural network")
    
    # Load model if specified
    if model_to_load:
        logger.info(f"📥 Loading model from: {model_to_load}")
        success = model.load_model(model_to_load)
        if success:
            model_loaded = True
            stats = model.get_statistics()
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"📊 Loaded stats: {stats['kills']}K/{stats['deaths']}D, Episodes: {stats['episode_count']}")
        else:
            logger.warning("⚠️ Model loading failed - continuing with fresh network")
    
    logger.info("🤖 ==========================================")
    
    # Create game interface
    game_interface = GameInterface(args.player_id, args.bot_name, model)
    
    try:
        logger.info("🔌 Connecting to Arena Battle Server...")
        logger.info("⏳ Server will automatically assign to PvP match")
        logger.info("🎯 Minimum 2 players required to start battle")
        logger.info("💾 Model will auto-save on significant improvements")
        logger.info("🛑 Press Ctrl+C to stop and save model")
        
        await game_interface.connect_and_play(
            host=args.host,
            port=args.port
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user - saving model...")
        await game_interface._save_model_on_exit()
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        if game_interface:
            await game_interface._save_model_on_exit()

if __name__ == "__main__":
    asyncio.run(main())