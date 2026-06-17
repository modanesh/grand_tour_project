#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pathlib
import numpy as np

# Suppress GUI warning for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Plot observations and actions diagnostic for a single robot.")
    parser.add_argument(
        "--robot-id", "-r",
        type=int,
        default=0,
        help="ID of the robot (env) to plot (default: 0)"
    )
    parser.add_argument(
        "--start-step", "-s",
        type=int,
        default=0,
        help="Starting step index within the robot's trajectory to plot (default: 0)"
    )
    parser.add_argument(
        "--num-steps", "-n",
        type=int,
        default=1000,
        help="Number of consecutive timesteps to plot (default: 1000)"
    )
    parser.add_argument(
        "--frequency", "-f",
        type=float,
        default=50.0,
        help="Simulation frequency in Hz for x-axis scaling (default: 50.0). Set to 0 to plot by index."
    )
    parser.add_argument(
        "--obs-path",
        type=str,
        default="generated_data/observations.csv",
        help="Path to observations CSV file"
    )
    parser.add_argument(
        "--action-path",
        type=str,
        default="generated_data/actions.csv",
        help="Path to actions CSV file"
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="generated_data/metadata.json",
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the output PNG plot. Defaults to generated_data/plot_robot_{robot_id}.png"
    )
    return parser.parse_args()

def load_data(file_path, start_row, nrows, expected_dim):
    """Load specific range of rows from a CSV file using pandas, falling back to standard csv module if pandas is not available."""
    print(f"Loading {nrows} rows starting from row {start_row} from {file_path}...")
    try:
        import pandas as pd
        # Using pandas with skiprows and nrows is extremely fast and memory-efficient
        df = pd.read_csv(file_path, header=None, skiprows=start_row, nrows=nrows)
        data = df.to_numpy()
    except ImportError:
        print("Pandas not found. Falling back to built-in csv module (this may be slower but is memory efficient)...")
        import csv
        import itertools
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            # Skip rows before start_row, then take nrows rows
            for row in itertools.islice(reader, start_row, start_row + nrows):
                if row:  # skip empty lines
                    data.append([float(x) for x in row])
        data = np.array(data)

    if data.shape[0] < nrows:
        print(f"Warning: Only loaded {data.shape[0]} rows (requested {nrows}). The file may have fewer steps than expected.")
    
    if data.shape[1] < expected_dim:
        raise ValueError(f"Error: Expected at least {expected_dim} columns in {file_path}, but found {data.shape[1]}.")

    return data

def main():
    args = parse_args()

    # Locate files relative to grand_tour_code root if not absolute
    script_dir = pathlib.Path(__file__).parent.parent.resolve()
    
    obs_path = pathlib.Path(args.obs_path)
    if not obs_path.is_absolute():
        obs_path = script_dir / obs_path
        
    action_path = pathlib.Path(args.action_path)
    if not action_path.is_absolute():
        action_path = script_dir / action_path
        
    metadata_path = pathlib.Path(args.metadata_path)
    if not metadata_path.is_absolute():
        metadata_path = script_dir / metadata_path

    # Parse metadata to check configuration
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            num_steps_per_env = metadata.get("num_steps_per_env", 1000)
            num_envs = metadata.get("num_envs", 4096)
            ordering = metadata.get("ordering", "robot-major")
            print(f"Metadata loaded: num_steps_per_env={num_steps_per_env}, num_envs={num_envs}, ordering={ordering}")
            
            if ordering != "robot-major":
                print(f"Warning: ordering is '{ordering}', expected 'robot-major'. The indexing might be incorrect.")
        except Exception as e:
            print(f"Warning: Failed to parse metadata file: {e}")
            num_steps_per_env = 1000
    else:
        print(f"Metadata file {metadata_path} not found. Assuming 1000 steps per env.")
        num_steps_per_env = 1000

    # Calculate global start row in the CSV file
    # In robot-major ordering, robot R's timesteps are stored contiguously:
    # Row R * num_steps_per_env to (R+1) * num_steps_per_env - 1
    start_row = args.robot_id * num_steps_per_env + args.start_step

    # Load observation and action data
    try:
        # observations.csv has 48 columns (36 obs + 12 prev_actions)
        obs_data = load_data(obs_path, start_row, args.num_steps, expected_dim=36)
        # actions.csv has 12 columns
        action_data = load_data(action_path, start_row, args.num_steps, expected_dim=12)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        sys.exit(1)

    # We only plot the first 36 dimensions from observations.csv
    obs = obs_data[:, :36]
    # We plot all 12 dimensions from actions.csv
    actions = action_data[:, :12]

    # Set up labels and color groups
    joint_names = [
        "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
        "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
        "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"
    ]

    obs_labels = (
        ['lin_vel_x', 'lin_vel_y', 'lin_vel_z',
         'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
         'grav_x',    'grav_y',    'grav_z',
         'cmd_vx',    'cmd_vy',    'cmd_yaw']
        + [f'jpos_{j}' for j in joint_names]
        + [f'jvel_{j}' for j in joint_names]
    )

    group_colors = (
        ['#1f77b4'] * 3    # lin_vel (blue)
        + ['#ff7f0e'] * 3  # ang_vel (orange)
        + ['#2ca02c'] * 3  # gravity (green)
        + ['#d62728'] * 3  # commands (red)
        + ['#9467bd'] * 12 # joint_pos (purple)
        + ['#8c564b'] * 12 # joint_vel (brown)
    )

    action_labels = [f'act_{j}' for j in joint_names]
    action_colors = ['#e377c2'] * 12  # pink

    # Prepare time/timestep axis
    actual_num_steps = min(len(obs), len(actions))
    if args.frequency > 0:
        t = (np.arange(actual_num_steps) + args.start_step) / args.frequency
        x_label = "Time (s)"
    else:
        t = np.arange(actual_num_steps) + args.start_step
        x_label = "Timestep"

    # Set up subplots: 8 rows by 6 columns (total 48 subplots)
    print("Creating diagnostic subplots...")
    fig, axes = plt.subplots(8, 6, figsize=(24, 18), dpi=150)
    axes = axes.flatten()

    # Plot 36 observation dimensions
    for i in range(36):
        axes[i].plot(t, obs[:actual_num_steps, i], linewidth=0.8, color=group_colors[i])
        axes[i].set_title(obs_labels[i], fontsize=8, pad=3)
        axes[i].set_xlabel(x_label, fontsize=7)
        axes[i].tick_params(labelsize=6)
        axes[i].grid(True, linewidth=0.3, alpha=0.5)

    # Plot 12 action dimensions
    for i in range(12):
        idx = 36 + i
        axes[idx].plot(t, actions[:actual_num_steps, i], linewidth=0.8, color=action_colors[i])
        axes[idx].set_title(action_labels[i], fontsize=8, pad=3)
        axes[idx].set_xlabel(x_label, fontsize=7)
        axes[idx].tick_params(labelsize=6)
        axes[idx].grid(True, linewidth=0.3, alpha=0.5)

    # Create the unified legend
    group_legend = [
        plt.Line2D([0], [0], color='#1f77b4', label='lin_vel'),
        plt.Line2D([0], [0], color='#ff7f0e', label='ang_vel'),
        plt.Line2D([0], [0], color='#2ca02c', label='gravity'),
        plt.Line2D([0], [0], color='#d62728', label='commands'),
        plt.Line2D([0], [0], color='#9467bd', label='joint_pos'),
        plt.Line2D([0], [0], color='#8c564b', label='joint_vel'),
        plt.Line2D([0], [0], color='#e377c2', label='actions'),
    ]
    fig.legend(
        handles=group_legend,
        loc='lower center',
        ncol=7,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01)
    )

    title_suffix = f" @ {args.frequency} Hz" if args.frequency > 0 else ""
    fig.suptitle(
        f"Robot {args.robot_id} Observations & Actions — Steps {args.start_step} to {args.start_step + actual_num_steps - 1}{title_suffix}",
        fontsize=14,
        y=0.98,
        fontweight='bold'
    )
    
    # Adjust layout to make room for legend and suptitle
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Determine output path
    if args.output:
        out_path = pathlib.Path(args.output)
    else:
        out_path = script_dir / f"generated_data/plot_robot_{args.robot_id}.png"

    # Create parent folder if it doesn't exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving plot to {out_path}...")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Plotting completed successfully!")

if __name__ == "__main__":
    main()
