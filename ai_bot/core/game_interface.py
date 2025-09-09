"""
Game Interface - Handles communication between AI models and game server
Provides a clean abstraction layer over gRPC protocol buffers
"""

import asyncio
import grpc
import logging
import sys
import os
import time
import math
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from proto import arena_pb2, arena_pb2_grpc
except ImportError:
    print("❌ Proto files not found! Run: python proto/generate.py")
    sys.exit(1)

from ..models.base_model import BaseAIModel, ObservationProcessor

logger = logging.getLogger(__name__)


class GameInterface:
    """Interface between AI models and game server"""
    
    def __init__(self, player_id: str, bot_name: str, model: BaseAIModel):
        self.player_id = player_id
        self.bot_name = bot_name
        self.model = model
        self.obs_processor = ObservationProcessor()
        
        # Connection state
        self.connected = False
        self.bot_id = None
        self.match_active = False
        
        # Game state tracking
        self.last_obs = None
        self.last_hp = 100.0
        self.last_enemy_hp = 100.0
        self.last_position = None
        
        # Enhanced movement system
        self.movement_bonus = 0.01
        self.stillness_penalty = -0.05
        self.smart_move_bonus = 0.02
        
        # Statistics
        self.episode_reward = 0.0
        self.waiting_start_time = None
        
        logger.info(f"🤖 Game Interface initialized: {self.bot_name}")
        logger.info(f"🧠 Using model: {self.model.get_model_info()['name']}")
    
    async def connect_and_play(self, host='localhost', port=50051):
        """Connect to server and start playing"""
        try:
            # Connect to server
            channel = grpc.aio.insecure_channel(f'{host}:{port}')
            stub = arena_pb2_grpc.ArenaBattleServiceStub(channel)
            
            # Register with server
            registration = arena_pb2.BotRegistration(
                player_id=self.player_id,
                bot_name=self.bot_name
            )
            
            logger.info(f"🤖 Registering with server: {self.bot_name}")
            response = await stub.RegisterBot(registration)
            
            if not response.success:
                logger.error(f"❌ Registration failed: {response.message}")
                return
            
            self.bot_id = response.bot_id
            self.connected = True
            
            logger.info(f"✅ Successfully registered with ID: {self.bot_id}")
            logger.info(f"📊 Status: {response.message}")
            
            # Check if waiting for players
            if "Waiting" in response.message:
                self.waiting_start_time = time.time()
                logger.info("⏳ Waiting for opponents to join battle...")
            
            try:
                # Start game loop
                await self._game_loop(stub)
            finally:
                # Save model on disconnect
                await self._save_model_on_exit()
                
        except Exception as e:
            logger.error(f"💥 Connection error: {e}")
        finally:
            if 'channel' in locals():
                await channel.close()
    
    async def _game_loop(self, stub):
        """Main game loop with AI decision making"""
        logger.info("🎮 Starting AI game loop...")
        
        action_queue = asyncio.Queue()
        
        # Start action sender task
        sender_task = asyncio.create_task(self._action_sender(action_queue))
        
        try:
            # Process observations from server
            async for observation in stub.PlayGame(self._action_generator(action_queue)):
                await self._process_observation(observation, action_queue)
                
        except Exception as e:
            logger.error(f"💥 Game loop error: {e}")
        finally:
            sender_task.cancel()
    
    async def _action_generator(self, action_queue):
        """Generate action stream for server"""
        try:
            while True:
                action = await action_queue.get()
                yield action
        except asyncio.CancelledError:
            pass
    
    async def _action_sender(self, action_queue):
        """Send default actions when AI is not active"""
        try:
            while True:
                # Send neutral action by default
                neutral_action = arena_pb2.Action(
                    thrust=arena_pb2.Vec2(x=0.0, y=0.0),
                    aim_angle=0.0,
                    fire=False
                )
                await action_queue.put(neutral_action)
                await asyncio.sleep(1/60)  # 60 FPS
                
        except asyncio.CancelledError:
            pass
    
    async def _process_observation(self, observation, action_queue):
        """Process observation and generate AI action"""
        try:
            # Check if match is active
            if observation.enemy_hp == 0 and observation.enemy_pos.x == 0:
                if not self.match_active:
                    # Still waiting for players
                    if self.waiting_start_time:
                        wait_time = time.time() - self.waiting_start_time
                        if wait_time % 10 < 0.1:  # Log every 10 seconds
                            logger.info(f"⏳ {self.bot_name} waiting for opponents... ({wait_time:.0f}s)")
                    return
                else:
                    # Match ended
                    self.match_active = False
                    logger.info("🏁 Combat engagement ended")
            else:
                # Match is active
                if not self.match_active:
                    self.match_active = True
                    if self.waiting_start_time:
                        wait_time = time.time() - self.waiting_start_time
                        logger.info(f"⚔️ {self.bot_name} combat started! (waited {wait_time:.1f}s)")
                        self.waiting_start_time = None
                    else:
                        logger.info(f"⚔️ {self.bot_name} joined ongoing combat!")
            
            # Only generate AI actions if match is active
            if self.match_active:
                await self._generate_ai_action(observation, action_queue)
            else:
                # Send neutral action while waiting
                neutral_action = arena_pb2.Action(
                    thrust=arena_pb2.Vec2(x=0.0, y=0.0),
                    aim_angle=0.0,
                    fire=False
                )
                await action_queue.put(neutral_action)
                
        except Exception as e:
            logger.error(f"💥 Observation processing error: {e}")
    
    async def _generate_ai_action(self, observation, action_queue):
        """Generate AI action from observation"""
        # Convert observation to standardized format
        obs_dict = self._observation_to_dict(observation)
        processed_obs = self.obs_processor.process(obs_dict)
        
        # Get action from AI model
        movement, aim, fire_action, log_prob = self.model.get_action(processed_obs)
        
        # Apply enhancements
        enhanced_movement = self._enhance_movement(movement, obs_dict)
        enhanced_aim = self._enhance_aiming(aim, obs_dict)
        enhanced_fire = self._enhance_firing(fire_action, obs_dict, enhanced_aim)
        
        # Track statistics
        if enhanced_fire:
            self.model.on_shot_fired()
        
        # Create action message
        action = arena_pb2.Action(
            thrust=arena_pb2.Vec2(
                x=float(enhanced_movement[0, 0].item()),
                y=float(enhanced_movement[0, 1].item())
            ),
            aim_angle=float(enhanced_aim[0, 0].item()),
            fire=bool(enhanced_fire[0].item() > 0.5)
        )
        
        # Calculate reward and handle learning
        reward = self._calculate_reward(obs_dict, enhanced_movement, enhanced_fire)
        done = obs_dict['self_hp'] <= 0
        
        if done:
            self.model.on_death()
            logger.info(f"💀 {self.bot_name} eliminated! Episode reward: {self.episode_reward:.2f}")
            self._reset_episode()
        
        # Update model with experience
        if self.last_obs is not None:
            experience = {
                'observation': self.last_obs,
                'action': (movement, aim, fire_action),
                'reward': reward,
                'next_observation': processed_obs,
                'done': done,
                'additional_info': {
                    'kills': self.model.kills,
                    'deaths': self.model.deaths
                }
            }
            
            metrics = self.model.learn_from_experience(experience)
            if metrics:
                logger.debug(f"📈 Training metrics: {metrics}")
        
        self.episode_reward += reward
        self.last_obs = processed_obs
        self.last_hp = obs_dict['self_hp']
        self.last_enemy_hp = obs_dict['enemy_hp']
        
        # Send action to server
        await action_queue.put(action)
    
    def _observation_to_dict(self, observation) -> Dict[str, Any]:
        """Convert protobuf observation to dictionary"""
        return {
            'tick': observation.tick,
            'self_pos': {'x': observation.self_pos.x, 'y': observation.self_pos.y},
            'self_hp': observation.self_hp,
            'enemy_pos': {'x': observation.enemy_pos.x, 'y': observation.enemy_pos.y},
            'enemy_hp': observation.enemy_hp,
            'bullets': [{'x': b.x, 'y': b.y} for b in observation.bullets],
            'walls': list(observation.walls),
            'has_line_of_sight': observation.has_line_of_sight,
            'arena_width': observation.arena_width,
            'arena_height': observation.arena_height
        }
    
    def _enhance_movement(self, movement, obs_dict):
        """Apply movement enhancements (wall avoidance, etc.)"""
        move_x = float(movement[0, 0].item())
        move_y = float(movement[0, 1].item())
        
        # Simple wall avoidance
        self_pos = obs_dict['self_pos']
        arena_width = obs_dict['arena_width']
        arena_height = obs_dict['arena_height']
        
        # Boundaries with safety margin
        margin = 50
        if self_pos['x'] < margin:
            move_x = max(move_x, 0.3)  # Push away from left wall
        if self_pos['x'] > arena_width - margin:
            move_x = min(move_x, -0.3)  # Push away from right wall
        if self_pos['y'] < margin:
            move_y = max(move_y, 0.3)  # Push away from top wall
        if self_pos['y'] > arena_height - margin:
            move_y = min(move_y, -0.3)  # Push away from bottom wall
        
        # Ensure minimum movement (anti-camping)
        movement_magnitude = math.sqrt(move_x**2 + move_y**2)
        if movement_magnitude < 0.3:
            # Add some random movement
            angle = np.random.uniform(0, 2 * np.pi)
            move_x += 0.2 * np.cos(angle)
            move_y += 0.2 * np.sin(angle)
        
        # Clamp to valid range
        move_x = np.clip(move_x, -1.0, 1.0)
        move_y = np.clip(move_y, -1.0, 1.0)
        
        # Create a new tensor for the enhanced movement
        enhanced_movement = movement.clone().fill_(0)
        enhanced_movement = enhanced_movement.squeeze(0)
        enhanced_movement[:2] = torch.tensor([move_x, move_y])
        enhanced_movement = enhanced_movement.unsqueeze(0)
        return enhanced_movement
    
    def _enhance_aiming(self, aim, obs_dict):
        """Apply aiming enhancements (smart targeting)"""
        enemy_pos = obs_dict['enemy_pos']
        self_pos = obs_dict['self_pos']
        
        if enemy_pos['x'] == 0 and enemy_pos['y'] == 0:
            return aim  # No enemy, keep current aim
        
        # Calculate angle to enemy
        dx = enemy_pos['x'] - self_pos['x']
        dy = enemy_pos['y'] - self_pos['y']
        if dx == 0 and dy == 0:
            return aim
        
        enemy_angle = math.atan2(dy, dx)
        current_aim = float(aim[0, 0].item())
        
        # Smooth adjustment toward enemy
        if obs_dict['has_line_of_sight']:
            angle_diff = enemy_angle - current_aim
            
            # Handle angle wrapping
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Apply partial correction
            enhanced_aim = current_aim + angle_diff * 0.7
        else:
            enhanced_aim = current_aim
        
        # Ensure valid range [0, 2π]
        enhanced_aim = enhanced_aim % (2 * math.pi)
        
        return aim.clone().fill_(enhanced_aim)
    
    def _enhance_firing(self, fire_action, obs_dict, aim):
        """Apply firing enhancements (smart shooting)"""
        enemy_pos = obs_dict['enemy_pos']
        self_pos = obs_dict['self_pos']
        
        if enemy_pos['x'] == 0 and enemy_pos['y'] == 0:
            return fire_action.clone().fill_(0)  # No enemy
        
        # Calculate distance
        dx = enemy_pos['x'] - self_pos['x']
        dy = enemy_pos['y'] - self_pos['y']
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Only fire if conditions are good
        should_fire = (
            obs_dict['has_line_of_sight'] and
            50 < distance < 500 and
            obs_dict['enemy_hp'] > 0
        )
        
        if should_fire:
            return fire_action  # Use AI's decision
        else:
            return fire_action.clone().fill_(0)  # Don't fire
    
    def _calculate_reward(self, obs_dict, movement, fired):
        """Calculate reward for current action"""
        reward = 0.0
        
        # Health change rewards
        current_hp = obs_dict['self_hp']
        if current_hp <= 0 and self.last_hp > 0:
            reward = -100.0  # Death penalty
        
        current_enemy_hp = obs_dict['enemy_hp']
        previous_enemy_hp = getattr(self, 'last_enemy_hp', 100.0)

        if previous_enemy_hp > 0 and current_enemy_hp <= 0:
            reward += 100.0  # Kill reward
            self.model.on_kill()
            logger.info(f"KILL! {self.bot_name} total kills: {self.model.kills}")
        
        # Movement rewards
        current_pos = (obs_dict['self_pos']['x'], obs_dict['self_pos']['y'])
        if self.last_position is not None:
            distance_moved = math.sqrt(
                (current_pos[0] - self.last_position[0])**2 + 
                (current_pos[1] - self.last_position[1])**2
            )
            
            if distance_moved > 2.0:
                reward += self.smart_move_bonus
            elif distance_moved > 1.0:
                reward += self.movement_bonus
            elif distance_moved < 0.5:
                reward += self.stillness_penalty
        
        self.last_position = current_pos
        self.last_enemy_hp = obs_dict['enemy_hp']
        return reward
    
    def _reset_episode(self):
        """Reset episode tracking"""
        stats = self.model.get_statistics()
        logger.info(f"📊 Episode complete - ACTUAL STATS: Kills={self.model.kills}, Deaths={self.model.deaths}")
        logger.info(f"📊 Episode complete - K/D: {stats['kd_ratio']:.2f}, Accuracy: {stats['accuracy']:.1f}%")
        
        self.episode_reward = 0.0
        self.last_hp = 100.0
        self.last_enemy_hp = 100.0
        self.last_position = None
    
    async def _save_model_on_exit(self):
        """Save model when exiting"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.player_id}_exit_{timestamp}.pth"
            models_dir = Path("models") / "checkpoints"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = models_dir / filename
            
            if self.model.save_model(str(filepath)):
                logger.info(f"💾 Model saved on exit: {filename}")
            
        except Exception as e:
            logger.error(f"💥 Failed to save model on exit: {e}")