import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class HoverDataset(Dataset):
    """PyTorch dataset for hover demonstrations"""
    def __init__(self, data):
        # Convert lists to numpy first, then to tensors (faster)
        obs_np = np.array(data['obs'], dtype=np.float32)
        actions_np = np.array(data['actions'], dtype=np.float32)
        
        self.obs = torch.from_numpy(obs_np)
        self.actions = torch.from_numpy(actions_np)
        
        print(f"📊 Dataset loaded: {len(self.obs)} samples")
        print(f"   Observation range: [{self.obs.min():.2f}, {self.obs.max():.2f}]")
        print(f"   Action range: [{self.actions.min():.2f}, {self.actions.max():.2f}]")
        print(f"   Action mean: {self.actions.mean(dim=0).numpy()}")
        print(f"   Action std: {self.actions.std(dim=0).numpy()}")
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

class PolicyNetwork(nn.Module):
    """Neural network that imitates PID controller"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def weighted_loss(pred, target, obs):
    """
    Weighted MSE that cares more about high angular velocity states.
    Think: "Mistakes matter more when you're spinning out of control"
    """
    # Base MSE
    mse = nn.MSELoss(reduction='none')(pred, target)
    
    # Extract angular velocities (last 3 dimensions of obs)
    ang_vel = obs[:, 10:13]
    ang_vel_magnitude = torch.norm(ang_vel, dim=1, keepdim=True)
    
    # Higher weight for high angular velocity (tumbling)
    # Weight ranges from 1.0 (stable) to 5.0 (tumbling)
    weights = 1.0 + 4.0 * torch.sigmoid(ang_vel_magnitude - 0.5)
    
    # Apply weights to each action dimension
    weighted_mse = mse * weights
    
    return weighted_mse.mean()

def train_imitation_policy(epochs=100, batch_size=64, learning_rate=3e-4):
    """Train neural network to imitate PID controller"""
    
    print("\n🎓 Starting imitation learning...")
    
    # Check if dataset exists
    if not os.path.exists('data/hover_dataset.npy'):
        print("❌ ERROR: data/hover_dataset.npy not found!")
        print("   Run collect_data.py first!")
        return None
    
    # Load dataset
    data = np.load('data/hover_dataset.npy', allow_pickle=True).item()
    
    # Validate dataset
    if len(data['obs']) == 0:
        print("❌ ERROR: Dataset is empty!")
        return None
    
    dataset = HoverDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    policy = PolicyNetwork(obs_dim=13, action_dim=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    best_loss = float('inf')
    
    print(f"\n🏋️  Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for obs, actions in loader:
            pred_actions = policy(obs)
            loss = weighted_loss(pred_actions, actions, obs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), 'data/pid_pretrained_policy_best.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} - Best: {best_loss:.6f}")
    
    # Save final model
    torch.save(policy.state_dict(), 'data/pid_pretrained_policy.pth')
    print(f"\n✅ Training complete!")
    print(f"💾 Final model saved to data/pid_pretrained_policy.pth")
    print(f"💾 Best model saved to data/pid_pretrained_policy_best.pth")
    print(f"📊 Final loss: {losses[-1]:.6f}")
    print(f"📊 Best loss: {best_loss:.6f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Imitation Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE Loss')
    plt.grid(True)
    plt.yscale('log')
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/training_loss.png')
    print(f"📊 Loss plot saved to data/training_loss.png")
    
    # Test the policy on a few samples
    print(f"\n🧪 Testing policy on sample data...")
    policy.eval()
    with torch.no_grad():
        # Get a batch of test data
        test_obs, test_actions = next(iter(loader))
        test_pred = policy(test_obs)
        
        # Show first 3 predictions vs actual
        for i in range(min(3, len(test_obs))):
            print(f"\nSample {i+1}:")
            print(f"   Predicted: {test_pred[i].numpy()}")
            print(f"   Actual:    {test_actions[i].numpy()}")
            print(f"   Error:     {torch.abs(test_pred[i] - test_actions[i]).numpy()}")
    
    return policy

if __name__ == "__main__":
    train_imitation_policy(epochs=100, batch_size=64, learning_rate=3e-4)