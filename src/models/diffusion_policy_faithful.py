"""
Faithful implementation of Diffusion Transformer Policy.
Based on diffusion_policy repo but without normalization.

Components:
- DDPMScheduler from diffusers
- LowdimMaskGenerator from diffusion_policy
- Proper conditional_sample inference
- Proper compute_loss training
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "diffusion_policy"))

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionTransformerPolicyFaithful(nn.Module):
    """
    Faithful diffusion policy implementation matching original repo.
    
    Key features (excludes normalization as requested):
    - DDPMScheduler for noise scheduling
    - LowdimMaskGenerator for conditioning masks
    - Proper conditional_sample for inference
    - Proper compute_loss for training
    """
    
    def __init__(
        self,
        obs_dim: int = 36,
        action_dim: int = 12,
        horizon: int = 8,
        n_layer: int = 6,
        n_head: int = 8,
        n_emb: int = 256,
        n_obs_steps: int = 1,
        n_action_steps: int = 1,
        num_inference_steps: int = 20,
        num_train_timesteps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = True,
        separate_goal_conditioning: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.prediction_type = prediction_type
        
        # Create the transformer model for noise prediction
        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=0,
            separate_goal_conditioning=separate_goal_conditioning,
        )
        
        # DDPM Scheduler from diffusers
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        
        # Mask generator for impainting/conditioning
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,  # obs_as_cond=True means obs is passed separately
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        
        self.num_inference_steps = num_inference_steps
        
    @property
    def diffusion_steps(self) -> int:
        """Number of diffusion timesteps (for training)."""
        return self.noise_scheduler.config.num_train_timesteps
    
    def get_device(self):
        """Get device of model parameters."""
        return next(self.parameters()).device
    
    def get_dtype(self):
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype
    
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample trajectory using DDPM, enforcing conditioning.
        
        Args:
            condition_data: Data to condition on (observations)
            condition_mask: Boolean mask indicating conditioned positions
            cond: Observation conditioning (when obs_as_cond=True)
            generator: Random generator for reproducibility
            
        Returns:
            Sampled trajectory of shape (B, T, action_dim)
        """
        device = condition_data.device
        dtype = condition_data.dtype
        
        # Initialize with noise
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        
        # Set scheduler timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)
        
        # Reverse diffusion process
        for t in self.noise_scheduler.timesteps:
            # 1. Apply conditioning: keep known observations fixed
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # 2. Predict noise residual
            model_output = self.model(trajectory, t, cond)
            
            # 3. Compute previous sample: x_t -> x_{t-1}
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory, generator=generator
            ).prev_sample
        
        # Final conditioning enforcement
        trajectory[condition_mask] = condition_data[condition_mask]
        
        return trajectory
    
    def predict_action(
        self,
        obs: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict action from observation.
        
        Args:
            obs: Observation tensor of shape (B, n_obs_steps, obs_dim)
            generator: Random generator for reproducibility
            
        Returns:
            Dict with keys:
                - 'action': (B, n_action_steps, action_dim) - predicted action
                - 'action_pred': (B, horizon, action_dim) - full trajectory
        """
        B, To, Do = obs.shape
        assert Do == self.obs_dim
        assert To == self.n_obs_steps
        
        T = self.horizon
        Da = self.action_dim
        device = obs.device
        dtype = obs.dtype
        
        # Extract conditioning
        cond = obs[:, :self.n_obs_steps, :]  # (B, n_obs_steps, obs_dim)
        
        # Build conditioning data and mask
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        # Run conditional sampling
        nsample = self.conditional_sample(
            cond_data, cond_mask, cond=cond, generator=generator
        )
        
        # Extract action for execution
        # Action is from To-1 to To-1 + n_action_steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = nsample[:, start:end, :]
        
        return {
            "action": action,
            "action_pred": nsample,
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            batch: Dict with keys:
                - 'obs': (B, To, obs_dim) - observations
                - 'action': (B, T, action_dim) - actions (trajectory)
                
        Returns:
            Scalar loss tensor
        """
        obs = batch["obs"]
        action = batch["action"]
        
        B = obs.shape[0]
        device = obs.device
        
        # Extract conditioning observations
        cond = obs[:, :self.n_obs_steps, :]  # (B, n_obs_steps, obs_dim)
        
        # Trajectory is just the action sequence (no concatenation since obs_as_cond)
        trajectory = action  # (B, T, action_dim)
        
        # Generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        
        # Sample noise and timesteps
        noise = torch.randn(trajectory.shape, device=device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()
        
        # Forward diffusion: add noise to clean trajectory
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # Apply conditioning: keep observations clean
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)
        
        # Determine target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")
        
        # Compute loss only on unmasked (non-conditioned) regions
        loss_mask = ~condition_mask
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        
        return loss
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (computes loss).
        
        Args:
            obs: (B, n_obs_steps, obs_dim)
            action: (B, horizon, action_dim)
            
        Returns:
            Loss tensor
        """
        batch = {"obs": obs, "action": action}
        return self.compute_loss(batch)
    
    def sample(self, obs: torch.Tensor, num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        Sample action trajectory from observation.
        
        Args:
            obs: (B, obs_dim) or (B, n_obs_steps, obs_dim) - observation
            num_inference_steps: Optional override for number of inference steps
            
        Returns:
            action_trajectory: (B, horizon, action_dim)
        """
        # Ensure obs has shape (B, n_obs_steps, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
        
        # Temporarily override num_inference_steps if provided
        original_steps = self.num_inference_steps
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps
        
        try:
            result = self.predict_action(obs)
            action_trajectory = result["action_pred"]
        finally:
            self.num_inference_steps = original_steps
        
        return action_trajectory
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Predict action from observation (numpy interface for compatibility).

        Args:
            obs: (B, obs_dim) or (obs_dim,) - observation

        Returns:
            action: (B, action_dim) or (action_dim,) - first action from trajectory
        """
        # Convert to tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.get_device())
        else:
            obs_tensor = obs.float().to(self.get_device())

        # Ensure batch dimension
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            action_traj = self.sample(obs_tensor)
            # action_traj: (B, horizon, action_dim)
            # Return first action from trajectory
            first_action = action_traj[:, 0, :]  # (B, action_dim)

        # Convert back to numpy
        result = first_action.cpu().numpy()

        # If input was 1D, return 1D
        if obs.ndim == 1:
            result = result[0]

        return result

    def get_optimizer(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        """Get optimizer with weight decay configuration."""
        return self.model.configure_optimizers(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
