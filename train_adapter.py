"""
Fine-tune adapter on Isaac Gym expert data.

Workflow:
1. Load pre-trained policy (trained on Grand Tour data)
2. Freeze policy weights, add lightweight adapter
3. Train adapter to map Isaac Gym obs → GT obs space
4. Evaluate adapted policy in Isaac Gym
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm

from adapter import ObservationAdapter, AdaptedPolicy
from utils import load_hdf5_dataset, compute_mean_std
from isaac_compatibility import make_actions_compatible
from grandtour_compatibility import unscale_observations
from eval_isaac_v2 import OnlineEval


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load pre-trained policy from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load workspace to get policy
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    workspace = BaseWorkspace.__new__(BaseWorkspace)
    workspace.load_payload(
        payload,
        exclude_keys=tuple(),
        include_keys=None,
        strict=False
    )

    # Get policy from workspace
    if hasattr(workspace, 'policy'):
        policy = workspace.policy
    else:
        raise ValueError("Checkpoint does not contain 'policy' attribute")

    policy.to(device)
    policy.eval()
    return policy


def train_adapter(
    policy,
    expert_data_path: str,
    isaac_obs_dim: int,
    gt_obs_dim: int,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    save_path: str = "adapter_checkpoint.pt",
):
    """Fine-tune adapter on expert Isaac Gym data."""

    print(f"\n{'='*60}")
    print("ADAPTER FINE-TUNING PHASE")
    print(f"{'='*60}")

    # Create adapter
    adapter = ObservationAdapter(isaac_obs_dim, gt_obs_dim, hidden_dim=256)
    adapter.to(device)

    # Create adapted policy (policy frozen, adapter trainable)
    adapted_policy = AdaptedPolicy(policy, adapter)
    adapted_policy.to(device)

    # Load expert data
    print(f"\nLoading expert data from: {expert_data_path}")
    dataset = load_hdf5_dataset(expert_data_path)
    expert_obs = torch.tensor(dataset['observations'], dtype=torch.float32, device=device)
    expert_actions = torch.tensor(dataset['actions'], dtype=torch.float32, device=device)

    print(f"Expert observations shape: {expert_obs.shape}")
    print(f"Expert actions shape: {expert_actions.shape}")

    # Normalize expert actions for loss computation
    action_mean = expert_actions.mean(dim=0)
    action_std = expert_actions.std(dim=0) + 1e-8
    expert_actions_norm = (expert_actions - action_mean) / action_std

    # Create data loader
    dataset_size = len(expert_obs)
    indices = torch.randperm(dataset_size)

    # Optimizer (only for adapter)
    optimizer = optim.Adam(adapter.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining adapter for {num_epochs} epochs with batch size {batch_size}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle indices
        perm = torch.randperm(dataset_size)

        for i in range(0, dataset_size, batch_size):
            batch_indices = perm[i:i+batch_size]
            batch_isaac_obs = expert_obs[batch_indices]  # (B, isaac_obs_dim)
            batch_expert_actions = expert_actions_norm[batch_indices]  # (B, action_dim)

            # Adapter transforms observation to GT space
            adapted_obs = adapter(batch_isaac_obs)

            # Frozen policy predicts actions from adapted observations
            with torch.no_grad():
                pred_actions_dict = policy.predict_action({'obs': adapted_obs})
                # Handle both dict and tensor returns
                if isinstance(pred_actions_dict, dict):
                    pred_actions = pred_actions_dict['action']
                else:
                    pred_actions = pred_actions_dict

            # Normalize predicted actions for loss
            pred_actions_norm = (pred_actions - action_mean) / action_std

            # Supervised loss: adapter should make policy predict expert actions
            loss = criterion(pred_actions_norm, batch_expert_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6e}")

    # Save adapter
    adapter_ckpt = {
        'adapter_state_dict': adapter.state_dict(),
        'isaac_obs_dim': isaac_obs_dim,
        'gt_obs_dim': gt_obs_dim,
        'config': {
            'hidden_dim': 256,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
        }
    }
    torch.save(adapter_ckpt, save_path)
    print(f"\nAdapter saved to: {save_path}")

    return adapted_policy, save_path


def evaluate_adapted_policy(
    adapted_policy,
    task_name: str,
    seed: int = 0,
    dataset_path: str = "offline_dataset_pp.hdf5",
):
    """Evaluate adapted policy in Isaac Gym."""

    print(f"\n{'='*60}")
    print("EVALUATION PHASE (Isaac Gym)")
    print(f"{'='*60}\n")

    # Use OnlineEval from eval_isaac_v2
    evaluator = OnlineEval(
        task_name=task_name,
        seed=seed,
        dataset_path=dataset_path,
        normalize=False,
        include_prev_actions=False,
    )

    # Evaluate
    eval_score, n_episodes, reward_terms, avg_ep_len, obs_stats = evaluator.eval_actor_isaac(
        adapted_policy,
        device="cuda"
    )

    print(f"\nEvaluation Results:")
    print(f"  Average reward: {eval_score:.4f}")
    print(f"  Episodes completed: {n_episodes}")
    print(f"  Average episode length: {avg_ep_len:.2f}")
    print(f"  Reward terms: {reward_terms}")

    return eval_score, n_episodes, reward_terms


def main():
    parser = argparse.ArgumentParser(description="Train adapter for cross-domain policy")
    parser.add_argument("--policy-checkpoint", type=str, required=True,
                        help="Path to pre-trained policy checkpoint (from diffuseloco.py)")
    parser.add_argument("--expert-data", type=str, default="expert_dataset.hdf5",
                        help="Path to expert Isaac Gym data (HDF5)")
    parser.add_argument("--isaac-obs-dim", type=int, default=36,
                        help="Isaac Gym observation dimension")
    parser.add_argument("--gt-obs-dim", type=int, default=36,
                        help="Grand Tour observation dimension")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for adapter training")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of adapter training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for adapter")
    parser.add_argument("--adapter-checkpoint", type=str, default="adapter_checkpoint.pt",
                        help="Path to save adapter checkpoint")
    parser.add_argument("--task", type=str, default="anymal_c_flat",
                        help="Isaac Gym task name for evaluation")
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="Seed for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Phase 1: Load pre-trained policy
    policy = load_policy(args.policy_checkpoint, device=args.device)

    # Phase 2: Train adapter on expert data
    adapted_policy, adapter_path = train_adapter(
        policy=policy,
        expert_data_path=args.expert_data,
        isaac_obs_dim=args.isaac_obs_dim,
        gt_obs_dim=args.gt_obs_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        save_path=args.adapter_checkpoint,
    )

    # Phase 3: Evaluate in Isaac Gym
    eval_score, n_episodes, reward_terms = evaluate_adapted_policy(
        adapted_policy=adapted_policy,
        task_name=args.task,
        seed=args.eval_seed,
        dataset_path=args.expert_data,
    )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Policy checkpoint: {args.policy_checkpoint}")
    print(f"Adapter checkpoint: {adapter_path}")
    print(f"Final eval score: {eval_score:.4f}")


if __name__ == "__main__":
    main()
