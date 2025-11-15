# train_imitation_policy.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class HoverDataset(Dataset):
    """PyTorch dataset for hover demonstrations (PWM targets in [0,1])."""
    def __init__(self, data):
        obs_np = np.array(data['obs'], dtype=np.float32)
        actions_np = np.array(data['actions'], dtype=np.float32)  # PWM [0,1]

        self.obs = torch.from_numpy(obs_np)
        self.actions = torch.from_numpy(actions_np)

        print(f"📊 Dataset: {len(self.obs)} samples")
        print(f"   Action (PWM) range: [{self.actions.min():.3f}, {self.actions.max():.3f}]")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

class PolicyNetwork(nn.Module):
    """Predicts 4× PWM ∈ [0,1] directly."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Sigmoid()  # ensure outputs ∈ [0,1]
        )

    def forward(self, x):
        return self.net(x)

def weighted_loss(pred, target, obs):
    """
    Weighted MSE — emphasize high angular velocity states (harder regimes).
    """
    mse = nn.MSELoss(reduction='none')(pred, target)
    ang_vel = obs[:, 10:13]
    ang_mag = torch.norm(ang_vel, dim=1, keepdim=True)
    weights = 1.0 + 4.0 * torch.sigmoid(ang_mag - 0.5)  # 1 → 5
    return (mse * weights).mean()

def train_imitation_policy(epochs=200, batch_size=64, learning_rate=3e-4):
    print("\n🎓 Imitation Learning — Predict PWM [0,1]")

    if not os.path.exists('data/hover_dataset.npy'):
        print("❌ data/hover_dataset.npy not found. Run collect_data.py first.")
        return None

    data = np.load('data/hover_dataset.npy', allow_pickle=True).item()
    if len(data['obs']) == 0:
        print("❌ Empty dataset.")
        return None

    dataset = HoverDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = PolicyNetwork(obs_dim=13, action_dim=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    losses = []
    best = float('inf')
    os.makedirs('data', exist_ok=True)

    print(f"\n🏋️ Training for {epochs} epochs...")
    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0
        n = 0
        for obs, actions in loader:
            pred = policy(obs)                # already in [0,1]
            loss = weighted_loss(pred, actions, obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(1, n)
        losses.append(avg)
        if avg < best:
            best = avg
            torch.save(policy.state_dict(), 'data/pid_pretrained_policy_best.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.6f} - Best: {best:.6f}")

    torch.save(policy.state_dict(), 'data/pid_pretrained_policy.pth')
    print("\n✅ Saved:")
    print("  • data/pid_pretrained_policy.pth")
    print("  • data/pid_pretrained_policy_best.pth")

    # Plot loss (log scale helps)
    plt.figure(figsize=(9,4))
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Imitation Loss (Weighted MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/training_loss.png')
    print("📊 Loss plot → data/training_loss.png")

    # Quick sanity test
    policy.eval()
    with torch.no_grad():
        obs, act = next(iter(loader))
        pred = policy(obs)
        for i in range(min(3, len(obs))):
            print(f"\nSample {i+1}:")
            print("  Pred:", pred[i].numpy())
            print("  True:", act[i].numpy())
            print("  Err :", np.abs(pred[i].numpy() - act[i].numpy()))

    return policy

if __name__ == "__main__":
    train_imitation_policy(epochs=200, batch_size=64, learning_rate=3e-4)
