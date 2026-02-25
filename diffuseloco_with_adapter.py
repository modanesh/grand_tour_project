"""
Unified training pipeline with adapter fine-tuning and Isaac Gym evaluation.

Workflow (loops continuously):
1. Train/update policy on Grand Tour data
2. Fine-tune adapter on Isaac Gym expert data
3. Evaluate adapted policy in Isaac Gym
4. Loop back to step 1

Usage:
python diffuseloco_with_adapter.py --config-name=anymal_diffusion_policy
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

try:
    from isaacgym.torch_utils import *
except:
    print("Isaac Gym Not Installed")

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

import hydra
from omegaconf import OmegaConf
import pathlib
import logging
from datetime import datetime

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from adapter import OutputDecoder, AdaptedPolicy
from eval_isaac_v2 import OnlineEval
from utils import load_hdf5_dataset

log = logging.getLogger(__name__)

# Register eval resolver for Hydra configs
OmegaConf.register_new_resolver("eval", eval, replace=True)


class AdapterTrainingWorkspace(BaseWorkspace):
    """
    Unified workspace that orchestrates:
    1. Grand Tour policy training
    2. Adapter fine-tuning on Isaac Gym expert data
    3. Isaac Gym evaluation
    4. Looping through phases
    """

    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        self.cycle = 0
        self.gt_policy = None
        self.output_decoder = None
        self.adapted_policy = None
        self.evaluator = None
        self.eval_history = []

    def _create_gt_workspace(self):
        """Create workspace for Grand Tour policy training."""
        log.info("=" * 60)
        log.info(f"[Cycle {self.cycle}] PHASE 1: Training on Grand Tour Data")
        log.info("=" * 60)

        # Instantiate the Grand Tour training workspace
        cls = hydra.utils.get_class(self.cfg.gt_workspace._target_)
        gt_workspace = cls(self.cfg.gt_workspace)

        return gt_workspace

    def _train_gt_policy(self, gt_workspace, num_steps: int = None):
        """Train/update policy on Grand Tour data."""
        log.info(f"Starting GT policy training (Cycle {self.cycle})...")

        # Run the GT training workspace
        # This uses the standard diffuseloco training procedure
        gt_workspace.run()

        # Get trained policy
        if hasattr(gt_workspace, 'policy'):
            self.gt_policy = gt_workspace.policy
            self.gt_policy.eval()
        else:
            raise ValueError("GT workspace doesn't have 'policy' attribute")

        log.info("✓ GT policy training complete")
        return self.gt_policy

    def _create_decoder(self, policy_output_dim: int, action_dim: int):
        """Create output decoder that maps policy outputs to Isaac Gym actions."""
        log.info(f"Creating output decoder: policy_output({policy_output_dim}D) → actions({action_dim}D)")
        self.output_decoder = OutputDecoder(policy_output_dim, action_dim, hidden_dim=256)
        self.output_decoder.to(self.cfg.device)

        self.adapted_policy = AdaptedPolicy(self.gt_policy, self.output_decoder)
        self.adapted_policy.to(self.cfg.device)

        log.info("✓ Output decoder created and frozen policy loaded")

    def _finetune_decoder(self):
        """Fine-tune output decoder on Isaac Gym expert data."""
        log.info("=" * 60)
        log.info(f"[Cycle {self.cycle}] PHASE 2: Fine-tuning Output Decoder on Expert Isaac Gym Data")
        log.info("=" * 60)

        expert_data_path = self.cfg.decoder.expert_data_path
        log.info(f"Loading expert data from: {expert_data_path}")

        dataset = load_hdf5_dataset(expert_data_path)
        expert_obs = torch.tensor(dataset['observations'], dtype=torch.float32, device=self.cfg.device)
        expert_actions = torch.tensor(dataset['actions'], dtype=torch.float32, device=self.cfg.device)

        log.info(f"Expert observations shape: {expert_obs.shape}")
        log.info(f"Expert actions shape: {expert_actions.shape}")

        # Normalize actions
        action_mean = expert_actions.mean(dim=0)
        action_std = expert_actions.std(dim=0) + 1e-8
        expert_actions_norm = (expert_actions - action_mean) / action_std

        # Setup optimizer
        import torch.optim as optim
        import torch.nn as nn

        optimizer = optim.Adam(self.output_decoder.parameters(), lr=self.cfg.decoder.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.decoder.num_epochs
        )
        criterion = nn.MSELoss()

        dataset_size = len(expert_obs)
        batch_size = self.cfg.decoder.batch_size

        log.info(f"Training decoder for {self.cfg.decoder.num_epochs} epochs")
        log.info(f"Batch size: {batch_size}, Dataset size: {dataset_size}")

        # Training loop
        for epoch in range(self.cfg.decoder.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            perm = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                batch_indices = perm[i : i + batch_size]
                batch_isaac_obs = expert_obs[batch_indices]
                batch_expert_actions = expert_actions_norm[batch_indices]

                # Get policy output for Isaac Gym observations
                with torch.no_grad():
                    policy_output = self.gt_policy.act_inference(batch_isaac_obs)

                # Decoder maps policy output to Isaac Gym actions
                pred_actions = self.output_decoder(policy_output)

                # Normalize predictions
                pred_actions_norm = (pred_actions - action_mean) / action_std

                # Loss: decoder should match expert actions
                loss = criterion(pred_actions_norm, batch_expert_actions)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.output_decoder.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            scheduler.step()

            log.info(
                f"Epoch {epoch+1}/{self.cfg.decoder.num_epochs} | "
                f"Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6e}"
            )

        log.info("✓ Output decoder fine-tuning complete")

    def _evaluate_isaac(self):
        """Evaluate adapted policy in Isaac Gym."""
        log.info("=" * 60)
        log.info(f"[Cycle {self.cycle}] PHASE 3: Evaluating in Isaac Gym")
        log.info("=" * 60)

        if self.evaluator is None:
            # NOTE: normalize=False during decoder training to keep observations raw
            self.evaluator = OnlineEval(
                task_name=self.cfg.eval.task_name,
                seed=self.cfg.eval.seed + self.cycle,  # Different seed each cycle
                dataset_path=self.cfg.decoder.expert_data_path,
                normalize=False,  # CRITICAL: Turn off normalization for decoder training
                include_prev_actions=self.cfg.eval.include_prev_actions,
            )

        eval_score, n_episodes, reward_terms, avg_ep_len, obs_stats = self.evaluator.eval_actor_isaac(
            self.adapted_policy,
            device=self.cfg.device,
        )

        log.info(f"\n{'─'*40}")
        log.info(f"Cycle {self.cycle} Evaluation Results:")
        log.info(f"{'─'*40}")
        log.info(f"  Average reward: {eval_score:.4f}")
        log.info(f"  Episodes completed: {n_episodes}")
        log.info(f"  Average episode length: {avg_ep_len:.2f}")
        log.info(f"  Reward terms: {reward_terms}")
        log.info(f"{'─'*40}\n")

        # Store in history
        self.eval_history.append({
            'cycle': self.cycle,
            'eval_score': eval_score,
            'n_episodes': n_episodes,
            'avg_ep_len': avg_ep_len,
        })

        return eval_score

    def _save_checkpoint(self, tag='latest'):
        """Save full checkpoint including policy and output decoder."""
        checkpoint = {
            'cycle': self.cycle,
            'gt_policy': self.gt_policy.state_dict() if self.gt_policy else None,
            'output_decoder': self.output_decoder.state_dict() if self.output_decoder else None,
            'eval_history': self.eval_history,
            'config': OmegaConf.to_container(self.cfg),
        }

        path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

        log.info(f"Checkpoint saved: {path}")
        return str(path.absolute())

    def run(self):
        """Main training loop."""
        log.info("\n" + "=" * 60)
        log.info("UNIFIED ADAPTER TRAINING PIPELINE")
        log.info("=" * 60 + "\n")

        num_cycles = self.cfg.num_cycles

        for self.cycle in range(num_cycles):
            log.info(f"\n{'#'*60}")
            log.info(f"## CYCLE {self.cycle + 1}/{num_cycles}")
            log.info(f"{'#'*60}\n")

            try:
                # Phase 1: Train GT policy
                gt_workspace = self._create_gt_workspace()
                self.gt_policy = self._train_gt_policy(gt_workspace)

                # Phase 2: Create/fine-tune decoder
                policy_output_dim = self.cfg.decoder.policy_output_dim
                action_dim = self.cfg.decoder.action_dim

                if self.output_decoder is None:
                    self._create_decoder(policy_output_dim, action_dim)

                self._finetune_decoder()

                # Phase 3: Evaluate in Isaac Gym
                eval_score = self._evaluate_isaac()

                # Save checkpoint
                self._save_checkpoint(tag=f'cycle_{self.cycle}')
                self._save_checkpoint(tag='latest')

            except Exception as e:
                log.error(f"Error in cycle {self.cycle}: {e}", exc_info=True)
                break

        log.info("\n" + "=" * 60)
        log.info("TRAINING COMPLETE")
        log.info("=" * 60)
        log.info(f"Total cycles completed: {self.cycle + 1}")
        log.info(f"Eval history: {self.eval_history}")


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath(
            "diffusion_policy", "diffusion_policy", "config"
        )
    ),
    config_name="anymal_diffusion_policy.yaml",
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # Merge in decoder-specific config
    if 'decoder' not in cfg:
        cfg.decoder = OmegaConf.create({
            'expert_data_path': 'expert_dataset.hdf5',
            'policy_output_dim': 12,  # Policy outputs action dimension
            'action_dim': 12,          # Isaac Gym action dimension
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-3,
            'hidden_dim': 256,
        })

    if 'eval' not in cfg:
        cfg.eval = OmegaConf.create({
            'task_name': 'anymal_d_flat',
            'seed': 0,
            'normalize': False,
            'include_prev_actions': False,
        })

    if 'num_cycles' not in cfg:
        cfg.num_cycles = 5

    if 'device' not in cfg:
        cfg.device = 'cuda'

    # Create workspace
    workspace = AdapterTrainingWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
