"""
INTERACTIVE TEST SCRIPT - PPO Drone Sphere Chasing (AirSim)
-----------------------------------------------------------
This script allows you to test your trained PPO model by simply pasting file paths.

✅ Supports both Gymnasium & Stable-Baselines3 VecEnv
✅ Safe GPU/CPU auto-detection
✅ Compatible with VecNormalize normalization
✅ Handles vectorized environment outputs correctly
"""

import os
import time
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_chase_env import DroneChaseEnv


def test_model(model_path, vecnormalize_path, episodes=5, render=False, sleep_time=0.05):
    """
    Test PPO model in AirSim environment.

    Args:
        model_path (str): Path to the PPO model (.zip)
        vecnormalize_path (str): Path to VecNormalize stats (.pkl)
        episodes (int): Number of episodes to run
        render (bool): Show step-by-step rewards
        sleep_time (float): Delay between steps for smoother visualization
    """
    print("\n" + "=" * 70)
    print("🧠  TESTING DRONE CHASING MODEL")
    print("=" * 70)
    print(f"📂 Model path: {model_path}")
    print(f"📁 VecNormalize path: {vecnormalize_path}")
    print(f"🎬 Episodes to run: {episodes}")
    print("=" * 70 + "\n")

    # --- Device setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # --- Create environment ---
    print("[1/4] Initializing AirSim environment...")
    env = DummyVecEnv([lambda: DroneChaseEnv()])

    # --- Load VecNormalize stats ---
    if os.path.exists(vecnormalize_path):
        print("[2/4] Loading normalization stats...")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ VecNormalize loaded successfully.\n")
    else:
        print("⚠️ VecNormalize file not found! Proceeding without normalization.\n")

    # --- Load model ---
    print("[3/4] Loading PPO model...")
    model = PPO.load(model_path, env=env, device=device)
    print("✓ Model loaded successfully.\n")

    # --- Run test episodes ---
    print("[4/4] Running test episodes...\n")
    total_rewards = []

    for ep in range(episodes):
        # FIXED: DummyVecEnv returns obs only, not (obs, info)
        obs = env.reset()
        done = [False]
        episode_reward = 0
        step = 0

        print(f"🚀 Starting Episode {ep + 1}/{episodes}")
        time.sleep(1)

        while not done[0]:
            # Get action from trained model (deterministic = no randomness)
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step += 1

            if render:
                print(f"  Step {step:03d} | Reward: {reward[0]:.3f}")

            # Delay for smoother visualization (especially in AirSim)
            time.sleep(sleep_time)

        print(f"✅ Episode {ep + 1} complete | Total Reward: {episode_reward:.2f}\n")
        total_rewards.append(episode_reward)

    print("=" * 70)
    print("🏁 TESTING SUMMARY")
    print("=" * 70)
    print(f"Total Episodes: {episodes}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Best Episode:   {np.max(total_rewards):.2f}")
    print(f"Worst Episode:  {np.min(total_rewards):.2f}")
    print("=" * 70 + "\n")

    env.close()
    print("✓ Environment closed safely.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared.\n")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                TEST TRAINED PPO MODEL IN AIRSIM                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")

    # --- Interactive inputs ---
    model_path = input("📂 Enter the full path to your PPO model (.zip): ").strip()
    while not os.path.exists(model_path):
        print("❌ File not found! Please re-enter a valid .zip file path.")
        model_path = input("📂 Enter the full path to your PPO model (.zip): ").strip()

    vecnormalize_path = input("📁 Enter the path to your VecNormalize stats (.pkl): ").strip()
    if not os.path.exists(vecnormalize_path):
        print("⚠️ VecNormalize file not found! Continuing without it.")

    episodes = input("🎬 Enter number of test episodes (default 5): ").strip()
    episodes = int(episodes) if episodes.isdigit() else 5

    render_input = input("👀 Show step-by-step reward updates? (y/n): ").strip().lower()
    render = render_input == "y"

    print("\nPreparing environment...")
    time.sleep(1)
    test_model(model_path, vecnormalize_path, episodes=episodes, render=render)
