"""Validation utilities for RMSE computation on held-out missions."""

import json
import numpy as np
import torch
from pathlib import Path

# Import dataloader
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataloader import GrandTourDataloader


def load_validation_missions(split_json_path="data/split.json"):
    """Load validation mission names from split.json."""
    with open(split_json_path, "r") as f:
        split = json.load(f)
    return split.get("validation_missions", [])


def _get_predictions(model, classifier, X_data, device, batch_size=64):
    """Get predictions from model for given observations.

    Args:
        model: Trained model
        classifier: Model type ("linear_regression", "ddpm", etc.)
        X_data: Observations array (N, 36)
        device: torch device
        batch_size: Batch size for inference

    Returns:
        torch.Tensor: Predictions (N, 12)
    """
    if classifier == "linear_regression":
        # sklearn LinearRegression
        predictions = model.predict(X_data)
        return torch.tensor(predictions, device=device, dtype=torch.float32)

    elif classifier == "random_forest":
        # sklearn RandomForestRegressor
        predictions = model.predict(X_data)
        return torch.tensor(predictions, device=device, dtype=torch.float32)

    elif classifier == "ddpm":
        # DiffusionTransformerPolicy - sample actions
        model.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_data), batch_size):
                batch_x = X_data[i : i + batch_size]
                batch_x_tensor = torch.tensor(
                    batch_x, device=device, dtype=torch.float32
                )

                # Sample action trajectory and take first action
                action_traj = model.sample(batch_x_tensor, num_inference_steps=20)
                # action_traj shape: (batch, horizon, action_dim)
                # Take first action from trajectory
                pred_actions = action_traj[:, 0, :]  # (batch, action_dim)
                all_preds.append(pred_actions.cpu().numpy())

        return torch.tensor(
            np.concatenate(all_preds, axis=0), device=device, dtype=torch.float32
        )

    else:
        # Generic diffusion model (DiffuseLocoModel)
        model.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_data), batch_size):
                batch_x = X_data[i : i + batch_size]
                batch_x_tensor = torch.tensor(
                    batch_x, device=device, dtype=torch.float32
                )

                # Predict next action
                t = torch.zeros(batch_x_tensor.shape[0], device=device)
                pred_actions = model(batch_x_tensor, t)
                all_preds.append(pred_actions.cpu().numpy())

        return torch.tensor(
            np.concatenate(all_preds, axis=0), device=device, dtype=torch.float32
        )


def compute_rmse_validation(
    model,
    classifier: str,
    device: torch.device,
    split_json_path: str = "data/split.json",
    frequency: int = 50,
    train_rmse: float = None,
):
    """Compute RMSE on validation missions from split.json.

    Args:
        model: Trained model (LinearRegression, DiffusionTransformerPolicy, etc.)
        classifier: Model type ("linear_regression", "ddpm", etc.)
        device: torch device
        split_json_path: Path to split.json containing validation missions
        frequency: Data sampling frequency
        train_rmse: Optional training RMSE to include in summary

    Returns:
        dict: RMSE statistics including per-dimension, per-mission, and overall RMSE
    """
    validation_missions = load_validation_missions(split_json_path)

    if not validation_missions:
        print("No validation missions found in split.json")
        return {}

    print(f"\n{'=' * 60}")
    print(f"Running RMSE validation on {len(validation_missions)} missions:")
    print(f"  {validation_missions}")
    print(f"{'=' * 60}")

    # Joint names for per-dimension logging
    joint_names = [
        "LF_HAA",
        "LH_HAA",
        "RF_HAA",
        "RH_HAA",
        "LF_HFE",
        "LH_HFE",
        "RF_HFE",
        "RH_HFE",
        "LF_KFE",
        "LH_KFE",
        "RF_KFE",
        "RH_KFE",
    ]

    # Track per-mission RMSE
    mission_rmse = {}
    all_predictions = []
    all_ground_truth = []

    # Compute RMSE for each mission individually
    for mission_name in validation_missions:
        print(f"\nProcessing mission: {mission_name}...")

        # Load single mission data
        dataloader = GrandTourDataloader(
            mission_names=[mission_name], frequency=frequency
        )
        X_val = dataloader.get_observations_isaac_lab_format()  # (N, 36)
        Y_val = dataloader.get_actions_isaac_lab_format()  # (N, 12) - next actions

        print(f"  Data shape: X={X_val.shape}, Y={Y_val.shape}")

        # Get predictions for this mission
        predictions = _get_predictions(model, classifier, X_val, device)
        Y_val_tensor = torch.tensor(Y_val, device=device, dtype=torch.float32)

        # Compute RMSE for this mission
        mse = torch.mean((predictions - Y_val_tensor) ** 2)
        rmse = torch.sqrt(mse).item()
        mission_rmse[mission_name] = rmse

        print(f"  Mission RMSE: {rmse:.6f}")

        # Accumulate for overall stats
        all_predictions.append(predictions)
        all_ground_truth.append(Y_val_tensor)

    # Concatenate all predictions and ground truth
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)

    print(f"\n{'=' * 60}")
    print(f"Combined validation data: {all_predictions.shape[0]} samples")
    print(f"{'=' * 60}")

    # Compute overall RMSE across all missions
    mse_per_dim = torch.mean((all_predictions - all_ground_truth) ** 2, dim=0)  # (12,)
    rmse_per_dim = torch.sqrt(mse_per_dim)  # (12,)
    overall_rmse = torch.sqrt(torch.mean(mse_per_dim)).item()

    # Build results dict
    results = {
        "validation/overall_rmse": overall_rmse,
    }

    # Add per-mission RMSE
    for mission_name, rmse in mission_rmse.items():
        results[f"validation/rmse_mission_{mission_name}"] = rmse

    # Add per-dimension RMSE with joint names
    for i, joint_name in enumerate(joint_names):
        results[f"validation/rmse_{joint_name}"] = rmse_per_dim[i].item()

    # Print summary
    print(f"\nValidation Results:")
    if train_rmse is not None:
        print(f"  Train RMSE: {train_rmse:.6f}")
        print(f"  Validation Overall RMSE: {overall_rmse:.6f}")
        print(f"  Generalization Gap: {overall_rmse - train_rmse:.6f}")
    else:
        print(f"  Overall RMSE: {overall_rmse:.6f}")
    print(f"\nPer-mission RMSE:")
    for mission_name, rmse in mission_rmse.items():
        print(f"  {mission_name}: {rmse:.6f}")
    print(f"\nPer-joint RMSE:")
    for i, joint_name in enumerate(joint_names):
        print(f"  {joint_name}: {rmse_per_dim[i].item():.6f}")
    print(f"{'=' * 60}\n")

    return results
