# ai_bot/models/hmoe_model.py  (refactored - single-file H-MoE + embedded experts)
from typing import Any, Dict
import math, random
from .base_model import BaseAIModel

# ---- Embedded Navigation expert (CNN-inspired) ----
class _NavCNN:
    def __init__(self):
        self.learning = {}
        self.steps_since_spawn = 0

    def _xy(self, p):
        if isinstance(p, dict):
            return float(p.get("x", 0.0)), float(p.get("y", 0.0))
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return float(p[0]), float(p[1])
        return (0.0, 0.0)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = observation if isinstance(observation, dict) else {}
        sx, sy = float(obs.get("self_pos", {"x":0,"y":0})["x"]), float(obs.get("self_pos", {"x":0,"y":0})["y"])
        enemy = obs.get("enemy_pos")
        bullets = obs.get("bullets", [])
        dodge = float(self.learning.get("dodge_bias", 0.25))

        # Aim
        aim_angle = 0.0
        if enemy is not None:
            ex, ey = self._xy(enemy)
            aim_angle = math.atan2(ey - sy, ex - sx) + float(self.learning.get("aim_correction", 0.0))

        # Move using a simple potential field: avoid bullets, drift toward enemy
        dx = dy = 0.0

        if enemy is None:
            # ưu tiên đẩy ra khỏi điểm spawn vài chục tick đầu
            steps = int(getattr(self, "_reposition_steps", 0) or 0)
            vec = getattr(self, "_reposition_vec", (0.0, 0.0))
            if steps > 0:
                dx += 0.6 * float(vec[0])
                dy += 0.6 * float(vec[1])
                self._reposition_steps = steps - 1
            else:
                t = int(getattr(self, "steps_since_spawn", 0) or 0)
                # wander nhẹ để không đứng yên
                dx += 0.45 * math.cos(0.12 * t)
                dy += 0.45 * math.sin(0.19 * t)

        # Avoid bullets
        for b in bullets[:24]:
            bx, by = self._xy(b)
            rx, ry = sx - bx, sy - by
            dist2 = rx*rx + ry*ry
            if dist2 < 1e-6: 
                continue
            w = min(1.0, 60_000.0 / dist2)
            dx += w * (rx)
            dy += w * (ry)

        # Drift toward enemy if any
        if enemy is not None:
            ex, ey = self._xy(enemy)
            dx += 0.15 * (ex - sx)
            dy += 0.15 * (ey - sy)

        # Normalize and clamp
        mag = math.hypot(dx, dy)
        if mag > 1.0:
            dx, dy = dx/mag, dy/mag
        dx *= (0.65 + 0.35*dodge)
        dy *= (0.65 + 0.35*dodge)

        fire = enemy is not None and mag < 1.0
        return {"move_x": float(dx), "move_y": float(dy), "aim_angle": float(aim_angle), "fire": bool(fire), "log_prob": 0.0}

# ---- Embedded Evasion expert (PPO-inspired with strafing memory) ----
class _EvaPPO:
    def __init__(self):
        self.learning = {}
        self.steps_since_spawn = 0

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = observation if isinstance(observation, dict) else {}
        sx, sy = float(obs.get("self_pos", {"x":0,"y":0})["x"]), float(obs.get("self_pos", {"x":0,"y":0})["y"])
        enemy = obs.get("enemy_pos")
        bullets = obs.get("bullets", [])

        # Aim
        aim_angle = 0.0
        if enemy is not None:
            ex, ey = float(enemy["x"]), float(enemy["y"])
            aim_angle = math.atan2(ey - sy, ex - sx) + float(self.learning.get("aim_correction", 0.0))

        # Strafe left-right based on 'strafe_amp' and 'dodge_bias' (learned after deaths)
        t = self.steps_since_spawn
        amp = float(self.learning.get("strafe_amp", 0.6))
        dodge = float(self.learning.get("dodge_bias", 0.5))
        # sinusoidal strafing; add small randomness
        dx = amp * math.cos(0.25 * t) + 0.05*(random.random()-0.5)
        dy = 0.1 * math.sin(0.17 * t)

        # If bullets incoming, bias perpendicular to their average velocity (dodge)
        if bullets:
            vx = sum(float(b.get("vx", 0.0)) for b in bullets[:8]) / min(8, len(bullets))
            vy = sum(float(b.get("vy", 0.0)) for b in bullets[:8]) / min(8, len(bullets))
            # perpendicular direction
            dx += dodge * (-vy) * 0.02
            dy += dodge * (vx) * 0.02

        # Slight pull toward enemy so we don't run away forever
        aim_angle = 0.0
        d = float("inf")          # <- khởi tạo mặc định
        if enemy is not None:
            ex, ey = float(enemy["x"]), float(enemy["y"])
            aim_angle = math.atan2(ey - sy, ex - sx) + float(self.learning.get("aim_correction", 0.0))
            rx, ry = ex - sx, ey - sy
            d = math.hypot(rx, ry) or 1.0

        # Clamp
        mag = math.hypot(dx, dy)
        if mag > 1.0:
            dx, dy = dx/mag, dy/mag

        fire = (d < 260.0) and (enemy is not None)
        return {"move_x": float(dx), "move_y": float(dy), "aim_angle": float(aim_angle), "fire": bool(fire), "log_prob": 0.0}

# ---- Embedded Combat expert (YOLO-inspired) ----
class _CombatYOLO:
    def __init__(self):
        self.learning = {}
        self.steps_since_spawn = 0

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = observation if isinstance(observation, dict) else {}
        sx, sy = float(obs.get("self_pos", {"x":0,"y":0})["x"]), float(obs.get("self_pos", {"x":0,"y":0})["y"])
        enemy = obs.get("enemy_pos")
        bullets = obs.get("bullets", [])

        # Aim at enemy (fallback if no detector)
        aim_angle = 0.0
        fire = False
        d = float("inf")  
        dx = dy = 0.0

        if enemy is not None:
            ex, ey = float(enemy["x"]), float(enemy["y"])
            rx, ry = ex - sx, ey - sy
            aim_angle = math.atan2(ry, rx) + float(self.learning.get("aim_correction", 0.0))
            d = math.hypot(rx, ry) or 1.0
            fire = (d < 300.0)

            # micro-strafing while shooting
            dx = 0.15 * (-math.sin(aim_angle))
            dy = 0.15 * ( math.cos(aim_angle))

        # Bullet avoidance lite
        for b in bullets[:6]:
            bx, by = float(b.get("x", 0.0)), float(b.get("y", 0.0))
            vx, vy = float(b.get("vx", 0.0)), float(b.get("vy", 0.0))
            # move perpendicular to bullet path
            dx += 0.01 * (-vy)
            dy += 0.01 * ( vx)

        mag = math.hypot(dx, dy)
        if mag > 1.0:
            dx, dy = dx/mag, dy/mag

        return {"move_x": float(dx), "move_y": float(dy), "aim_angle": float(aim_angle), "fire": bool(fire), "log_prob": 0.0}

# ---- HMoe container that gates between experts ----
class HMoeModel(BaseAIModel):
    def __init__(self):
        super().__init__(model_name="hmoe")
        self.nav = _NavCNN()
        self.eva = _EvaPPO()
        self.combat = _CombatYOLO()

    def _context(self, obs: Dict[str, Any]) -> Dict[str, float]:
        sx, sy = float(obs.get("self_pos", {"x":0,"y":0})["x"]), float(obs.get("self_pos", {"x":0,"y":0})["y"])
        enemy = obs.get("enemy_pos")
        if enemy is not None:
            ex, ey = float(enemy.get("x", 0.0)), float(enemy.get("y", 0.0))
            d_enemy = math.hypot(ex - sx, ey - sy)
        else:
            d_enemy = 1e6
        # bullet time-to-impact estimate
        tti_min = 1e9
        for b in obs.get("bullets", [])[:20]:
            bx, by = float(b.get("x", 0.0)), float(b.get("y", 0.0))
            vx, vy = float(b.get("vx", 0.0)), float(b.get("vy", 0.0))
            rx, ry = sx - bx, sy - by
            vproj = (vx*rx + vy*ry) / (math.hypot(vx, vy) + 1e-6)
            t = (math.hypot(rx, ry) / (abs(vproj) + 1e-3)) if vproj > 0 else 1e9
            if t < tti_min: tti_min = t
        return {"d_enemy": d_enemy, "tti_min": tti_min}
    def process(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        HMoe không cần tiền xử lý — trả về nguyên dict.
        Đặt no-op để tương thích với lời gọi trong act().
        """
        return observation if isinstance(observation, dict) else {}

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = observation if isinstance(observation, dict) else {}
        self._last_obs = obs
        self.steps_since_spawn = getattr(self, "steps_since_spawn", 0) + 1

        ctx = self._context(obs)
        d, tti = ctx["d_enemy"], ctx["tti_min"]

        # Gating rules:
        #   imminent bullet -> Evasion, close enemy -> Combat, else -> Navigation
        if tti < 80.0:
            a = self.eva.act(obs)
        elif d < 240.0:
            a = self.combat.act(obs)
        else:
            a = self.nav.act(obs)
        # Failsafe: nếu output gần như 0 -> cưỡng bức wander để không đứng yên
        dx = float(a.get("move_x", 0.0))
        dy = float(a.get("move_y", 0.0))
        if dx*dx + dy*dy < 1e-4:
            theta = float(getattr(self, "_wander_dir", 0.0))
            theta += (random.random() - 0.5) * 0.2  # hơi đổi hướng
            self._wander_dir = theta
            a["move_x"] = 0.5 * math.cos(theta)
            a["move_y"] = 0.5 * math.sin(theta)
            # không bắn khi chưa xác định địch
            if obs.get("enemy_pos") is None:
                a["fire"] = False
        return a 

    # === Abstract interface implementations ===
    def _initialize_model(self,**kwargs):
        """Initialize internal HMoe state (no external weights)."""
        self.learning = getattr(self, 'learning', {})
        self.update_count = getattr(self, 'update_count', 0)
        self.steps_since_spawn = 0

    def get_action(self, observation, deterministic: bool = False):
        """Adapter to BaseAIModel interface. Accepts dict or tensor-like. Returns: movement, aim, fire, dummy_logprob        """
        # We expect BotClient to pass a dict; if it's tensor, fallback to neutral.
        try:
            obs = observation if isinstance(observation, dict) else {}
            a = self.act(obs)
            # Convert to tensors-like matches PPOModel usage: movement(1,2), aim(1,1), fire(1,1)
            import torch
            movement = torch.tensor([[float(a.get('move_x', 0.0)), float(a.get('move_y', 0.0))]], dtype=torch.float32)
            aim = torch.tensor([[float(a.get('aim_angle', 0.0))]], dtype=torch.float32)
            fire = torch.tensor([[1.0 if a.get('fire', False) else 0.0]], dtype=torch.float32)
            log_prob = torch.tensor([[0.0]], dtype=torch.float32)
            return movement, aim, fire, log_prob
        except Exception:
            # Safe fallback
            import torch
            return (torch.zeros((1,2), dtype=torch.float32),
                    torch.zeros((1,1), dtype=torch.float32),
                    torch.zeros((1,1), dtype=torch.float32),
                    torch.zeros((1,1), dtype=torch.float32))

    def learn_from_experience(self, experience_data):
        """Lightweight online update: adjust dodge/strafe after death/survival."""
        # Expect dict with keys like 'event': 'death' or 'kill', 'accuracy', 'wall_hits'
        if not isinstance(experience_data, dict):
            return False
        ev = experience_data.get('event')
        self.learning = getattr(self, 'learning', {})
        if ev == 'death':
            self.learning['strafe_amp'] = float(self.learning.get('strafe_amp', 0.6)) * 1.05
            self.learning['dodge_bias'] = float(self.learning.get('dodge_bias', 0.5)) * 1.05
        elif ev == 'kill':
            self.learning['strafe_amp'] = float(self.learning.get('strafe_amp', 0.6)) * 0.98
        if 'aim_error' in experience_data:
            corr = float(self.learning.get('aim_correction', 0.0))
            self.learning['aim_correction'] = corr * 0.9 + 0.1 * (-experience_data['aim_error'])
        self.update_count = getattr(self, 'update_count', 0) + 1
        return True

    def save_model(self, filepath: str) -> bool:
        """Save lightweight HMoe state to a torch checkpoint."""
        try:
            import torch
            ckpt = {
                'model_name': 'HMoeModel',
                'learning': getattr(self, 'learning', {}),
                'update_count': getattr(self, 'update_count', 0),
            }
            torch.save(ckpt, filepath)
            return True
        except Exception:
            return False

    def load_model(self, filepath: str) -> bool:
        """Load lightweight HMoe state from a torch checkpoint."""
        try:
            import torch, os
            if not os.path.exists(filepath):
                return False
            ckpt = torch.load(filepath, map_location='cpu')
            self.learning = ckpt.get('learning', {})
            self.update_count = ckpt.get('update_count', 0)
            return True
        except Exception:
            return False
