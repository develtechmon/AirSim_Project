# train_sphere_chaser.py
"""
Interactive PPO Trainer for Drone–Sphere Chase (AirSim)
-------------------------------------------------------
User options:
  [1] Train new model (custom configuration)
  [2] Test existing model
  [3] Continue training existing model
"""

import os
import time
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from airsim_sphere_env import AirSimSphereChaseEnv

MODELS_DIR = Path("models/sphere_chaser")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VEC_PATH = MODELS_DIR / "vecnormalize.pkl"


# ---------- Utilities ----------
def make_env(visualize=False, seed=42):
    def _init():
        env = AirSimSphereChaseEnv(visualize=visualize, seed=seed)
        return Monitor(env)
    return _init


def build_env(n_envs=1, visualize=False, seed=42):
    vec_env = DummyVecEnv([make_env(visualize=visualize, seed=seed + i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)
    return vec_env


def save_vecnorm(vec_env):
    vec_env.save(str(VEC_PATH))


def load_vecnorm(vec_env, training=False):
    if VEC_PATH.exists():
        vec_env.load(str(VEC_PATH))
        vec_env.training = training
        vec_env.norm_reward = training
    return vec_env


def list_models():
    models = sorted(MODELS_DIR.glob("**/*.zip"))
    if not models:
        print("No models found.")
        return []
    for i, m in enumerate(models):
        print(f"[{i}] {m}")
    return models


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


# ---------- Menu ----------
def main_menu():
    print("\n==============================")
    print(" AIRSIM DRONE-SPHERE TRAINER ")
    print("==============================")
    print("[1] Train New Model")
    print("[2] Test Existing Model")
    print("[3] Continue Training Model")
    print("==============================")
    return input("Select option (1/2/3): ").strip()


# ---------- Training ----------
def train_new():
    print("\n--- New Training Configuration ---")
    timesteps = input("Training timesteps (default 300000): ").strip()
    timesteps = int(timesteps) if timesteps else 300_000

    lr = input("Learning rate (default 3e-4): ").strip()
    lr = float(lr) if lr else 3e-4

    n_envs = input("Number of parallel envs (default 4): ").strip()
    n_envs = int(n_envs) if n_envs else 4

    env = build_env(n_envs=n_envs, visualize=False, seed=42)
    model = PPO("MlpPolicy", env, learning_rate=lr, verbose=1, batch_size=256, n_steps=2048)

    ckpt_dir = MODELS_DIR / f"ppo_{timestamp()}"
    ckpt_dir.mkdir(exist_ok=True)
    callback = CheckpointCallback(save_freq=50_000 // n_envs, save_path=str(ckpt_dir), name_prefix="ckpt")

    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save(str(ckpt_dir / "final"))
    save_vecnorm(env)
    env.close()
    print(f"\n✅ Training complete! Model saved at: {ckpt_dir}\n")


# ---------- Testing ----------
def test_existing():
    models = list_models()
    if not models:
        return
    idx = int(input("Select model index: ").strip() or 0)
    model_path = models[idx]

    env = build_env(n_envs=1, visualize=True)
    env = load_vecnorm(env, training=False)
    model = PPO.load(str(model_path), env=env)

    episodes = int(input("Number of test episodes (default 5): ") or 5)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                hits = info[0].get("hits", 0) if isinstance(info, list) else info.get("hits", 0)
                print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}, hits={hits}")
                break
    env.close()


# ---------- Continue ----------
def continue_training():
    models = list_models()
    if not models:
        return
    idx = int(input("Select checkpoint index: ").strip() or 0)
    model_path = models[idx]

    extra_steps = int(input("Additional timesteps (default 200000): ").strip() or 200_000)

    env = build_env(n_envs=4, visualize=False)
    env = load_vecnorm(env, training=True)

    model = PPO.load(str(model_path), env=env)
    ckpt_dir = Path(model_path).parent
    callback = CheckpointCallback(save_freq=50_000, save_path=str(ckpt_dir), name_prefix="ckpt_cont")

    model.learn(total_timesteps=extra_steps, callback=callback, progress_bar=True)
    model.save(str(ckpt_dir / "final_cont"))
    save_vecnorm(env)
    env.close()
    print(f"\n✅ Continued training complete! Updated model saved at: {ckpt_dir}\n")


# ---------- Entry ----------
if __name__ == "__main__":
    choice = main_menu()
    if choice == "1":
        train_new()
    elif choice == "2":
        test_existing()
    elif choice == "3":
        continue_training()
    else:
        print("Invalid option.")
