"""
Lightweight output decoder for domain adaptation between Grand Tour and Isaac Gym.
Maps policy outputs to Isaac Gym action space.
"""
import torch
import torch.nn as nn


class OutputDecoder(nn.Module):
    """
    Simple MLP that maps policy outputs to Isaac Gym actions.

    Flow: policy_output → decoder → isaac_actions
    """

    def __init__(self, policy_output_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.policy_output_dim = policy_output_dim
        self.action_dim = action_dim

        # Simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(policy_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, policy_output: torch.Tensor) -> torch.Tensor:
        """
        Map policy output to Isaac Gym action space.

        Args:
            policy_output: (B, policy_output_dim) Policy network output

        Returns:
            isaac_actions: (B, action_dim) Isaac Gym actions
        """
        return self.net(policy_output)


class AdaptedPolicy(nn.Module):
    """
    Wraps a frozen policy with an output decoder for cross-domain inference.

    Flow:
    Isaac Gym obs → Frozen policy → policy output → Output decoder → Isaac Gym actions

    During fine-tuning: only decoder is trainable.
    During eval: decoder + policy work together.
    """

    def __init__(self, policy, output_decoder: OutputDecoder):
        super().__init__()
        self.policy = policy  # Frozen during decoder fine-tuning
        self.output_decoder = output_decoder  # Only thing trained during decoder phase

        # Freeze policy weights
        for param in self.policy.parameters():
            param.requires_grad = False

        # Decoder starts trainable (unfrozen)
        for param in self.output_decoder.parameters():
            param.requires_grad = True

    def forward(self, isaac_obs_dict):
        """
        Isaac Gym obs dict → Policy → Decoder → actions
        """
        isaac_obs = isaac_obs_dict['obs']  # (B, obs_dim)

        # Get policy output via predict_action — expects (B, n_obs_steps, obs_dim)
        with torch.no_grad():
            obs_seq = isaac_obs.unsqueeze(1).expand(-1, self.policy.n_obs_steps, -1)
            result = self.policy.predict_action({'obs': obs_seq})
            policy_output = result['action'][:, 0, :]  # (B, 12)

        # Decoder maps policy output to Isaac Gym actions
        isaac_actions = self.output_decoder(policy_output)

        return isaac_actions

    def act_inference(self, isaac_obs: torch.Tensor) -> torch.Tensor:
        """
        Direct inference interface for evaluation (called by eval_isaac_v2.py).

        Isaac obs tensor → Policy → Decoder → Isaac actions tensor

        Args:
            isaac_obs: (B, obs_dim) Raw Isaac Gym observations

        Returns:
            isaac_actions: (B, action_dim) Isaac Gym action predictions
        """
        # Get policy output via predict_action — expects (B, n_obs_steps, obs_dim)
        with torch.no_grad():
            obs_seq = isaac_obs.unsqueeze(1).expand(-1, self.policy.n_obs_steps, -1)
            result = self.policy.predict_action({'obs': obs_seq})
            policy_output = result['action'][:, 0, :]  # (B, 12)

        # Decoder maps policy output to Isaac Gym actions
        isaac_actions = self.output_decoder(policy_output)

        return isaac_actions

    def freeze_decoder(self):
        """Freeze decoder for policy-only inference."""
        for param in self.output_decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """Unfreeze decoder for fine-tuning."""
        for param in self.output_decoder.parameters():
            param.requires_grad = True
