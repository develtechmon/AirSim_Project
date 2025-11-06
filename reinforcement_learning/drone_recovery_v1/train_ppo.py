# train_ppo.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from airsim_env import AirSimDroneEnv


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Curriculum Manager: Stage Control Logic                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
class CurriculumManager:
    def __init__(self):
        self.stage = 1
        self.history = {1: [], 2: [], 3: []}
        self.threshold = {1: -20.0, 2: 0.0, 3: 5.0}  # thresholds tuned for dense reward
        self.min_episodes = 20

    def record_episode(self, stage, reward):
        self.history[stage].append(float(reward))

    def get_current_stage(self):
        return self.stage

    def should_advance(self):
        recent = self.history[self.stage][-self.min_episodes:]
        if len(recent) < self.min_episodes:
            return False
        avg_reward = np.mean(recent)
        return (avg_reward > self.threshold[self.stage]) and (self.stage < 3)

    def advance_stage(self):
        self.stage = min(3, self.stage + 1)
        return self.stage


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Neural Feature Extractor (shared π & V trunk)                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=256)
        n_in = int(observation_space.shape[0])
        self.net = nn.Sequential(
            nn.LayerNorm(n_in),
            nn.Linear(n_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Progress Callback with TensorBoard + Curriculum Updates             ║
# ╚══════════════════════════════════════════════════════════════════════╝
class DetailedProgressCallback(BaseCallback):
    def __init__(self, curriculum_manager, env, total_timesteps, writer, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum_manager
        self.env = env
        self.writer = writer
        self.total_timesteps = total_timesteps

        self.current_timestep = 0
        self.episode_reward = 0.0
        self.episode_len = 0
        self.episode_count = 0

        self.recent_rewards = []
        self.stage_start_time = time.time()

    def _on_step(self):
        """Triggered every simulation step"""
        self.current_timestep += 1
        reward = self.locals['rewards'][0]
        self.episode_reward += reward
        self.episode_len += 1

        done = self.locals['dones'][0]
        if done:
            info = self.locals['infos'][0]
            reason = info.get("terminal_reason", "other")
            stage = self.curriculum.get_current_stage()

            self.episode_count += 1
            self.recent_rewards.append(self.episode_reward)
            self.curriculum.record_episode(stage, self.episode_reward)

            # TensorBoard logging
            self.writer.add_scalar(f"Stage{stage}/EpisodeReward", self.episode_reward, self.episode_count)
            self.writer.add_scalar(f"Stage{stage}/EpisodeLength", self.episode_len, self.episode_count)

            # Print summary
            avg10 = np.mean(self.recent_rewards[-10:]) if len(self.recent_rewards) >= 10 else np.mean(self.recent_rewards)
            print(f"\n📊 EP {self.episode_count} | STAGE {stage} | {100*self.current_timestep/self.total_timesteps:.2f}% done")
            print(f"  Reward: {self.episode_reward:>8.2f} | Avg(10): {avg10:>8.2f} | Len: {self.episode_len}")
            print(f"  Termination: {reason}\n")

            # Curriculum promotion
            if self.curriculum.should_advance():
                old_stage = stage
                new_stage = self.curriculum.advance_stage()
                self.env.set_stage(new_stage)
                print(f"🎉 Curriculum advance: Stage {old_stage} → {new_stage}")
                self.model.save(f"data/ppo_stage{old_stage}_checkpoint.zip")
                self.recent_rewards = []

            # Reset per-episode stats
            self.episode_reward = 0.0
            self.episode_len = 0

            # Adjust hover bias dynamically based on stability
            if avg10 > -40 and self.env.hover_bias > 0.0:
                self.env.hover_bias = max(0.0, self.env.hover_bias - 0.005)
                print(f"⚙️  Hover bias reduced → {self.env.hover_bias:.3f}")

        return True

    def _on_training_end(self):
        total_time = time.time() - self.stage_start_time
        print(f"\n🏁 Training Complete — {self.episode_count} episodes in {total_time/60:.1f} min")
        self.writer.close()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Load pretrained imitation weights                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
def load_pretrained_policy(model):
    try:
        device = model.policy.device
        path = 'data/pid_pretrained_policy_best.pth' if os.path.exists('data/pid_pretrained_policy_best.pth') else 'data/pid_pretrained_policy.pth'
        state_dict = torch.load(path, map_location=device)

        feat = model.policy.features_extractor
        model_dict = feat.state_dict()
        loadable = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(loadable)
        feat.load_state_dict(model_dict, strict=False)
        print(f"✅ Loaded imitation features from {path}")
    except Exception as e:
        print(f"⚠️ Imitation weight load failed: {e}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Full Training Pipeline                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
def train_full_curriculum():
    print("=" * 80)
    print("🎓 FULL CURRICULUM — PPO POST-IMPACT RECOVERY")
    print("=" * 80)
    print("Stages: 1=Hover | 2=Disturbance | 3=Recovery\n")

    # Initialize environment and curriculum
    curriculum = CurriculumManager()
    env = AirSimDroneEnv(stage=1, hover_bias=0.05)
    total_timesteps = 500_000

    # TensorBoard setup
    os.makedirs("data", exist_ok=True)
    writer = SummaryWriter(log_dir="ppo_logs")

    # PPO architecture
    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs={},
        net_arch=dict(pi=[256], vf=[256]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        tensorboard_log="./ppo_logs/",
        policy_kwargs=policy_kwargs
    )

    load_pretrained_policy(model)
    print("✅ PPO initialized with imitation-trained feature extractor.")

    callback = DetailedProgressCallback(curriculum, env, total_timesteps, writer)
    print(f"\n🚀 Starting training for {total_timesteps:,} steps...\n")

    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_stage = curriculum.get_current_stage()
    model.save(f"data/ppo_stage{final_stage}_final.zip")
    torch.save(model.policy.state_dict(), f"data/ppo_stage{final_stage}_weights.pth")

    print("\n✅ Training complete!")
    print(f"Final stage: {final_stage}")
    print(f"Saved: data/ppo_stage{final_stage}_final.zip")


if __name__ == "__main__":
    train_full_curriculum()
