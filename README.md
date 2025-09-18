# Arena Battle Game - AI Combat Training Platform

A 2D top-down battle arena game where AI bots learn real-time combat strategies through reinforcement learning. The system features server-managed room-based multiplayer architecture with advanced PPO (Proximal Policy Optimization) AI training.

## Overview

Arena Battle Game is a sophisticated AI training platform that combines real-time physics simulation with machine learning. AI bots develop combat strategies including wall avoidance, smart aiming, tactical movement, and firing efficiency through continuous learning in PvP battles.

**Key Features:**
- Real-time multiplayer combat with room-based system
- Advanced PPO reinforcement learning with enhanced reward system  
- Smart wall avoidance and tactical movement AI
- Auto-save model checkpoints with performance tracking
- Live visualization with pygame UI
- Configurable arena layouts with custom obstacles
- gRPC-based client-server architecture
- Comprehensive JSON logging system

## Project Structure

```
arena_battle_game/
├── requirements.txt              # Dependencies
├── README.md                    # This file
├── rooms.json                   # Room configurations
│
├── proto/                       # Protocol Buffer definitions
│   ├── arena.proto             # gRPC service definitions
│   ├── arena_pb2.py            # Python classes
│   ├── arena_pb2_grpc.py       # gRPC stubs
│
├── game_server/                 # Centralized game engine
│   ├── main.py                 # Server entry point
│   │
│   ├── engine/                 # Core game logic
│   │   ├── game_state.py       # Game state management
│   │   └── physics.py          # Physics engine
│   │
│   ├── networking/             # Network communication
│   │   ├── server.py           # gRPC server implementation
│   │   └── room_manager.py     # Room-based matchmaking
│   │
│   ├── logging/                # JSON logging system
│   │   └── json_logger.py      # Server-side data logging
│   │
│   └── ui/                     # Visualization
│       └── renderer.py         # Real-time pygame rendering
│
├── ai_bot/                     # AI client implementation
│   ├── main.py                 # Bot entry point
│   │
│   ├── models/                 # Neural networks
│   │   ├── base_model.py       # Abstract AI model interface
│   │   ├── network.py          # PPO network architecture
│   │   └── ppo_model.py        # Modular PPO implementation
│   │
│   ├── training/               # Learning algorithms
│   │   ├── ppo.py              # PPO trainer with tactical rewards
│   │   └── buffer.py           # Experience replay buffer
│   │
│   └── client/                 # Connection logic
│       └── bot_client.py       # Enhanced gRPC client
│
├── models/                     # Model storage
│   ├── checkpoints/            # Auto-saved training checkpoints
│   └── backups/               # Manual model backups
│
└── logs/                      # Server logging
    └── server_grpc_data/      # JSON logs of all gRPC data
```

## Environment Setup

### Prerequisites
- Python 3.8+
- Virtual environment support (venv/conda)
- Git

### 1. Create Virtual Environment

**Using venv:**
```bash
# Create virtual environment
python -m venv arena_battle

# Activate (Windows)
arena_battle\Scripts\activate

# Activate (Linux/Mac)
source arena_battle/bin/activate
```

**Using conda:**
```bash
# Create conda environment
conda create -n arena_battle python=3.9

# Activate environment
conda activate arena_battle
```

### 2. Clone Repository
```bash
git clone <repository-url>
cd arena_battle_game
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Protocol Buffers
```bash
python proto/generate.py
```

## Room Configuration

Edit `rooms.json` to configure battle arenas:

```json
{
    "room_001": {
        "password": "abc123",
        "max_players": 4,
        "arena": {
            "width": 800,
            "height": 600,
            "obstacles": [
                {"x": 200, "y": 200, "width": 50, "height": 100},
                {"x": 550, "y": 300, "width": 100, "height": 50}
            ]
        }
    }
}
```

## Running the System

### Step 1: Start Game Server
```bash
python -m game_server.main
```

**Server Features:**
- Real-time physics simulation at 60 FPS
- Live pygame visualization with room switching (R key)
- Speed control: 1x, 2x, 4x, 10x training speeds
- JSON logging of all gRPC communications
- Multi-room support with custom arena layouts

### Step 2: Connect AI Bots

**Basic Connection:**
```bash
python -m ai_bot.main --player-id player001 --room-id room_001 --room-password abc123
```

**With Model Loading:**
```bash
python -m ai_bot.main --player-id player002 --room-id room_001 --room-password abc123 --auto-load
```

**Training Parameters:**
```bash
python -m ai_bot.main \
  --player-id advanced_bot \
  --room-id room_002 \
  --room-password abc456 \
  --model-path ./models/checkpoints/trained_model.pth
```

### Step 3: Monitor Training

**List Available Models:**
```bash
python -m ai_bot.main --player-id player001 --list-models
```

**Server Controls:**
- `1,2,3,4` - Adjust training speed (1x to 10x)
- `R` - Cycle through room views
- `D` - Toggle debug visualization
- `Click` - Select bot to follow
- `ESC` - Shutdown server

## Game Mechanics

### Combat System
- **Health:** 100 HP, 25 damage per hit (4 hits to eliminate)
- **Movement:** Continuous thrust control with 200 px/s max speed
- **Shooting:** 0.3s cooldown, 400 px/s bullet speed
- **Respawn:** 1s delay at random position with 1s invulnerability

### AI Learning Features
- **Wall Avoidance:** Dynamic collision detection with danger zones
- **Smart Aiming:** Predictive targeting with line-of-sight awareness  
- **Tactical Movement:** Anti-camping with strategic positioning
- **Firing Efficiency:** Conservative ammunition management
- **Real-time Learning:** Model updates after each death/kill event

### Room System
- **Minimum Players:** 2 players required to start combat
- **Waiting State:** Solo players receive stable waiting observations
- **Room Capacity:** Configurable max players per room
- **Password Protection:** Secure room access control

## AI Architecture

### PPO Network Design
- **Observation Space:** 48-dimensional normalized feature vectors
- **Action Space:** 
  - Movement: Continuous 2D thrust (-1 to 1)
  - Aiming: Continuous angle (0 to 2π radians) 
  - Firing: Discrete boolean decision
- **Network Layers:** 3-layer fully connected (128 hidden units)
- **Training:** Actor-critic with GAE advantage estimation

### Enhanced Reward System
```python
rewards = {
    'kill_enemy': +100,
    'death': -100, 
    'movement': +0.01,
    'stillness': -0.05,
    'wall_collision': -0.1,
    'smart_firing': +0.005,
    'wasted_ammo': -0.01
}
```

### Auto-Save System
- **Time-based:** Every 5 minutes by default
- **Performance-based:** On K/D ratio improvements
- **Event-triggered:** Every 10 deaths for progress tracking
- **Model Management:** Automatic cleanup of old checkpoints

### Performance Optimization
- **Headless Mode:** `python -m game_server.main --no-ui`
- **Speed Scaling:** Use 4x-10x speed for rapid training iterations
- **Batch Training:** Run multiple bot instances simultaneously
- **Model Checkpointing:** Regular saves prevent training loss

## Logging and Analytics

### JSON Logs
All gRPC communications are logged to `logs/server_grpc_data/`:
- Bot registrations and disconnections
- All observations sent to bots  
- All actions received from bots
- Game events (kills, deaths, respawns)
- Match events and room assignments
- Performance metrics and errors

### Model Statistics
Each saved model includes comprehensive metrics:
- Kill/Death ratios and accuracy percentages
- Training episodes and total rewards
- Wall collision counts and movement patterns
- Firing efficiency and tactical performance

## Troubleshooting

### Common Issues

**Connection Refused:**
- Ensure server is running on correct port (50051)
- Check firewall settings for localhost connections

**Model Loading Failed:**
- Verify model file path exists
- Check model compatibility with current network architecture

**Room Access Denied:**
- Confirm room ID and password in rooms.json
- Ensure room isn't at maximum capacity

### Performance Issues
- Reduce training speed multiplier (1x-2x)
- Enable headless mode for CPU optimization
- Check system resources during intensive training

## Contributing

The codebase follows a modular architecture supporting:
- Custom AI algorithm implementations
- New arena layouts and game mechanics
- Enhanced visualization and UI features
- Advanced logging and analytics systems

## License

This project is developed for educational and research purposes in reinforcement learning and multi-agent systems.
