"""
TRAIN SCRIPT - Imitation Learning + PPO Fine-tuning
Author: Lukas the Big Boss + GPT-5
"""

import os, sys, time, torch, argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from airsim_recovery_env_rate import PostImpactRecoveryEnvRate, TeacherPD

# --------------------------
# Dataset
# --------------------------
class HoverDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.obs = data['obs']
        self.act = data['act']

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return torch.tensor(self.obs[i], dtype=torch.float32), torch.tensor(self.act[i], dtype=torch.float32)


# --------------------------
# Imitation Model (same MLP as PPO)
# --------------------------
class PolicyNet(nn.Module):
    def __init__(self, in_dim=36, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------
# Collect Data from Teacher
# --------------------------
def collect_data(episodes=10, out="data_hover.npz"):
    env = PostImpactRecoveryEnvRate(stage="hover", enable_logging=False)
    teacher = TeacherPD(alt_target=env.spawn_altitude)

    obs_buf, act_buf = [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        t0 = time.time()

        while not done and step < 400:
            act = teacher.get_action(obs)
            obs_buf.append(obs)
            act_buf.append(act)
            obs, _, done, _, _ = env.step(act)
            step += 1

        print(f"[collect] Episode {ep+1}/{episodes} | steps={step}, dt={time.time()-t0:.1f}s")

    env.close()
    np.savez(out, obs=np.array(obs_buf), act=np.array(act_buf))
    print(f"✅ Saved dataset: {out} | {len(obs_buf)} samples")


# --------------------------
# Pretrain Policy (Imitation)
# --------------------------
def pretrain(data_path="data_hover.npz", save_path="pid_clone.pt", epochs=20):
    ds = HoverDataset(data_path)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for e in range(epochs):
        running = 0
        for obs, act in dl:
            opt.zero_grad()
            pred = net(obs)
            loss = loss_fn(pred, act)
            loss.backward()
            opt.step()
            running += loss.item() * len(obs)
        print(f"[epoch {e+1}/{epochs}] loss={running/len(ds):.6f}")

    torch.save(net.state_dict(), save_path)
    print(f"✅ Saved imitation weights: {save_path}")
    return save_path


# --------------------------
# PPO Fine-tuning
# --------------------------
def train_ppo(init_model="pid_clone.pt", timesteps=100000):
    def make_env():
        return Monitor(PostImpactRecoveryEnvRate(stage="disturbance", enable_logging=False))

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, n_steps=2048,
                batch_size=256, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.005, vf_coef=0.5, policy_kwargs=policy_kwargs)

    # Load imitation weights
    policy = model.policy.mlp_extractor.policy_net
    sd = torch.load(init_model)
    policy.load_state_dict(sd, strict=False)
    print("✅ PPO initialized from imitation weights")

    log_path = "./logs_ppo"
    os.makedirs(log_path, exist_ok=True)
    model.set_logger(configure(log_path, ["tensorboard", "stdout"]))

    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("ppo_finetuned")
    print("✅ PPO fine-tuned model saved")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--train-rl", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="data_hover.npz")
    parser.add_argument("--data", type=str, default="data_hover.npz")
    parser.add_argument("--save", type=str, default="pid_clone.pt")
    parser.add_argument("--init", type=str, default="pid_clone.pt")
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()

    if args.collect:
        collect_data(args.episodes, args.out)
    elif args.pretrain:
        pretrain(args.data, args.save)
    elif args.train_rl:
        train_ppo(args.init, args.timesteps)
