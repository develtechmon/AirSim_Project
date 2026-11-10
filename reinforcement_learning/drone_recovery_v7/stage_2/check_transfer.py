import torch, numpy as np, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class Dummy(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (13,), np.float32)
    action_space = spaces.Box(-5, 5, (3,), np.float32)
    def reset(self, seed=None, options=None): return np.zeros(13, np.float32), {}
    def step(self, a): return np.zeros(13, np.float32), 0.0, False, False, {}

model = PPO("MlpPolicy", DummyVecEnv([Dummy]), device="cpu",
            policy_kwargs=dict(net_arch=[256, 256, 128]))
print("PPO activation:", model.policy.activation_fn)
print("PPO layer names:")
for k, v in model.policy.state_dict().items():
    print("  ", k, tuple(v.shape))

# exact same mapping as train_stage2_disturbance.py
bc = torch.load("./models/hover_policy_best.pth", map_location="cpu")
mapped = {}
for k, p in bc.items():
    n, t = int(k.split('.')[1]), k.split('.')[2]
    mapped[f'features_extractor.mlp.{n//2}.{t}' if n < 6 else f'action_net.{t}'] = p
res = model.policy.load_state_dict(mapped, strict=False)
print("\nIGNORED (not loaded):", res.unexpected_keys)