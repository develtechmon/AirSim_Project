"""
MODEL LOADER FOR TRAINED POLICIES
==================================
Loads your trained Stage 1, 2, 3 models for deployment.

Handles:
- Stage 1: Behavioral Cloning (PyTorch)
- Stage 2: PPO (Stable-Baselines3)
- Stage 3: PPO (Stable-Baselines3)
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle


class HoverPolicy(nn.Module):
    """Stage 1 Behavioral Cloning Policy (13 → 256 → 256 → 128 → 3)"""
    
    def __init__(self, state_dim=13, action_dim=3):
        super(HoverPolicy, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def predict(self, state):
        """Predict action for a single state"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            action = self.network(state).squeeze(0).numpy()
            # Clip to safe range
            action = np.clip(action, -5.0, 5.0)
            return action


class ModelLoader:
    """
    Unified model loader for all stages
    
    Usage:
        loader = ModelLoader()
        
        # Stage 1 (Behavioral Cloning)
        policy = loader.load_stage1('./models/hover_policy_best.pth')
        
        # Stage 2 (PPO with normalization)
        policy = loader.load_stage2(
            './models/hover_disturbance_policy.zip',
            './models/hover_disturbance_vecnormalize.pkl'
        )
        
        # Stage 3 (PPO with normalization)
        policy = loader.load_stage3(
            './models/gated_curriculum_policy.zip',
            './models/gated_curriculum_vecnormalize.pkl'
        )
    """
    
    def __init__(self):
        self.device = torch.device("cpu")
    
    def load_stage1(self, model_path):
        """
        Load Stage 1 Behavioral Cloning model
        
        Args:
            model_path: Path to .pth file
        
        Returns:
            policy object with predict(state) method
        """
        print(f"Loading Stage 1 model: {model_path}")
        
        policy = HoverPolicy(state_dim=13, action_dim=3)
        policy.load_state_dict(torch.load(model_path, map_location=self.device))
        policy.eval()
        
        print("✅ Stage 1 model loaded")
        print(f"   Architecture: 13 → 256 → 256 → 128 → 3")
        print(f"   Outputs: [vx, vy, vz] velocity commands")
        
        return policy
    
    def load_stage2(self, model_path, vecnorm_path=None):
        """
        Load Stage 2 PPO model
        
        Args:
            model_path: Path to .zip file
            vecnorm_path: Path to vecnormalize .pkl file
        
        Returns:
            (model, vecnorm) tuple
        """
        print(f"Loading Stage 2 model: {model_path}")
        
        # Load model
        model = PPO.load(model_path, device=self.device)
        
        # Load VecNormalize stats if available
        vecnorm = None
        if vecnorm_path:
            print(f"Loading normalization stats: {vecnorm_path}")
            vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: None]))
            vecnorm.training = False
            vecnorm.norm_reward = False
        
        print("✅ Stage 2 model loaded")
        print(f"   Architecture: 13 → 256 → 256 → 128 → 3")
        print(f"   Outputs: [vx, vy, vz] velocity commands")
        print(f"   Normalization: {'Enabled' if vecnorm else 'Disabled'}")
        
        return model, vecnorm
    
    def load_stage3(self, model_path, vecnorm_path=None):
        """
        Load Stage 3 PPO model (same as Stage 2)
        
        Args:
            model_path: Path to .zip file
            vecnorm_path: Path to vecnormalize .pkl file
        
        Returns:
            (model, vecnorm) tuple
        """
        print(f"Loading Stage 3 model: {model_path}")
        
        model = PPO.load(model_path, device=self.device)
        
        vecnorm = None
        if vecnorm_path:
            print(f"Loading normalization stats: {vecnorm_path}")
            vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: None]))
            vecnorm.training = False
            vecnorm.norm_reward = False
        
        print("✅ Stage 3 model loaded")
        print(f"   Architecture: 13 → 256 → 256 → 128 → 3")
        print(f"   Outputs: [vx, vy, vz] velocity commands")
        print(f"   Normalization: {'Enabled' if vecnorm else 'Disabled'}")
        
        return model, vecnorm


def get_observation_from_vehicle(client):
    """
    Get 13-dimensional observation from ArduPilot
    
    Same format as AirSim training:
        [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    
    Args:
        client: ArduPilotInterface instance
    
    Returns:
        np.array of shape (13,)
    """
    state = client.getMultirotorState()
    
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    ori = state.kinematics_estimated.orientation
    ang_vel = state.kinematics_estimated.angular_velocity
    
    obs = np.array([
        pos.x_val, pos.y_val, pos.z_val,           # Position (NED)
        vel.x_val, vel.y_val, vel.z_val,           # Velocity (NED)
        ori.w_val, ori.x_val, ori.y_val, ori.z_val, # Orientation (quaternion)
        ang_vel.x_val, ang_vel.y_val, ang_vel.z_val # Angular velocity
    ], dtype=np.float32)
    
    return obs


def predict_with_normalization(model, vecnorm, obs):
    """
    Predict action with VecNormalize
    
    Args:
        model: PPO model
        vecnorm: VecNormalize object (or None)
        obs: Raw observation (13,)
    
    Returns:
        action: [vx, vy, vz] velocity commands
    """
    # Normalize observation if vecnorm is provided
    if vecnorm:
        obs_normalized = vecnorm.normalize_obs(obs.reshape(1, -1))
    else:
        obs_normalized = obs.reshape(1, -1)
    
    # Predict
    action, _ = model.predict(obs_normalized, deterministic=True)
    
    # Clip to safe range
    action = np.clip(action, -5.0, 5.0)
    
    return action.flatten()