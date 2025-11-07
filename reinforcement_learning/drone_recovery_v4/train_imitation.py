"""
BEHAVIORAL CLONING TRAINING
============================
Trains a neural network to imitate the PID expert using supervised learning.

This is MUCH faster than pure RL:
- Training time: 20-30 minutes
- Expected success: 90%+
- Data: Your collected expert demonstrations

python train_imitation.py --dataset ./demonstrations/expert_demonstrations.pkl

TIMING:

Takes about 20-30 minutes to complete
Shows progress every 10 epochs
You can watch it train in real-time

SUCCESS CRITERIA:
After training finishes, check:

Final Val Loss < 0.05 → Excellent! (95%+ success)
Final Val Loss 0.05-0.10 → Good (85-95% success)
Final Val Loss > 0.10 → Needs more data

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import argparse
from pathlib import Path
import time


class ExpertDataset(Dataset):
    """Dataset of expert demonstrations"""
    
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class HoverPolicy(nn.Module):
    """
    Neural network policy for hover control
    
    Architecture: 10 → 256 → 256 → 128 → 3
    - Input: State (10 dimensions: position, velocity, orientation)
    - Output: Action (3 dimensions: vx, vy, vz commands)
    """
    
    def __init__(self, state_dim=10, action_dim=3):
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
        """Predict action for a single state (for deployment)"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            return self.network(state).squeeze(0).numpy()


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for states, actions in train_loader:
        states, actions = states.to(device), actions.to(device)
        
        # Forward pass
        predicted_actions = model(states)
        loss = criterion(predicted_actions, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for states, actions in val_loader:
            states, actions = states.to(device), actions.to(device)
            predicted_actions = model(states)
            loss = criterion(predicted_actions, actions)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main(args):
    print("\n" + "="*70)
    print("🎓 BEHAVIORAL CLONING TRAINING")
    print("="*70 + "\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("[1/5] Loading dataset...")
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)
    
    states = data['states']
    actions = data['actions']
    
    print(f"   Total samples: {len(states):,}")
    print(f"   State dimension: {states.shape[1]}")
    print(f"   Action dimension: {actions.shape[1]}")
    print(f"   Mean episode reward: {np.mean(data['rewards']):.1f}")
    print()
    
    # Create dataset and split
    print("[2/5] Creating train/validation split...")
    dataset = ExpertDataset(states, actions)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Training samples: {train_size:,}")
    print(f"   Validation samples: {val_size:,}")
    print()
    
    # Create model
    print("[3/5] Creating model...")
    model = HoverPolicy(state_dim=states.shape[1], action_dim=actions.shape[1])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Architecture: {states.shape[1]} → 256 → 256 → 128 → {actions.shape[1]}")
    print()
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("[4/5] Training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print()
    
    print("="*70)
    print("EPOCH | TRAIN LOSS | VAL LOSS | TIME")
    print("="*70)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {epoch_time:4.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/hover_policy_best.pth")
    
    total_time = time.time() - start_time
    
    print("="*70)
    print()
    
    # Save final model
    print("[5/5] Saving model...")
    torch.save(model.state_dict(), f"{args.save_dir}/hover_policy_final.pth")
    
    # Save model info
    model_info = {
        'state_dim': states.shape[1],
        'action_dim': actions.shape[1],
        'architecture': '10 → 256 → 256 → 128 → 3',
        'total_params': total_params,
        'best_val_loss': best_val_loss,
        'training_samples': train_size,
        'training_time_minutes': total_time / 60
    }
    
    with open(f"{args.save_dir}/model_info.pkl", 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"   ✅ Best model: {args.save_dir}/hover_policy_best.pth")
    print(f"   ✅ Final model: {args.save_dir}/hover_policy_final.pth")
    print(f"   ✅ Model info: {args.save_dir}/model_info.pkl")
    print()
    
    # Final statistics
    print("="*70)
    print("📊 TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Total Training Time: {total_time/60:.1f} minutes")
    print()
    
    # Estimate success rate
    if best_val_loss < 0.05:
        success_estimate = "95%+"
    elif best_val_loss < 0.10:
        success_estimate = "85-95%"
    elif best_val_loss < 0.20:
        success_estimate = "70-85%"
    else:
        success_estimate = "< 70% (may need more data)"
    
    print(f"📈 Estimated Success Rate: {success_estimate}")
    print()
    print("✅ Next step: Run test_hover_policy.py to evaluate!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hover policy via behavioral cloning")
    
    parser.add_argument('--dataset', type=str, default='./demonstrations/expert_demonstrations.pkl',
                        help='Path to expert demonstrations')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU available')
    
    args = parser.parse_args()
    
    main(args)