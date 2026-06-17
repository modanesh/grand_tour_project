import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-idx", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="/home/exps/Projects/rohan/grand_tour_project/grand_tour_code/generated_data/spaces_plot.png")
    args = parser.parse_args()

    data_dir = "/home/exps/Projects/rohan/grand_tour_project/grand_tour_code/generated_data"
    obs_path = os.path.join(data_dir, "observations.csv")
    act_path = os.path.join(data_dir, "actions.csv")
    meta_path = os.path.join(data_dir, "metadata.json")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    num_steps = metadata.get("num_steps_per_env", 500)
    start_row = args.robot_idx * num_steps

    obs_df = pd.read_csv(obs_path, header=None, skiprows=start_row, nrows=num_steps)
    act_df = pd.read_csv(act_path, header=None, skiprows=start_row, nrows=num_steps)

    obs = obs_df.values
    actions = act_df.values

    joint_names = [
        "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
        "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
        "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"
    ]

    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    steps = np.arange(num_steps)

    axs[0, 0].plot(steps, obs[:, 0], label="Lin Vel X")
    axs[0, 0].plot(steps, obs[:, 1], label="Lin Vel Y")
    axs[0, 0].plot(steps, obs[:, 2], label="Lin Vel Z")
    axs[0, 0].plot(steps, obs[:, 3], label="Ang Vel X", linestyle="--")
    axs[0, 0].plot(steps, obs[:, 4], label="Ang Vel Y", linestyle="--")
    axs[0, 0].plot(steps, obs[:, 5], label="Ang Vel Z", linestyle="--")
    axs[0, 0].set_title(f"Base Velocities (Robot {args.robot_idx})")
    axs[0, 0].set_ylabel("Velocity (m/s or rad/s)")
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].grid(True)

    axs[0, 1].plot(steps, obs[:, 6], label="Grav X")
    axs[0, 1].plot(steps, obs[:, 7], label="Grav Y")
    axs[0, 1].plot(steps, obs[:, 8], label="Grav Z")
    axs[0, 1].plot(steps, obs[:, 9], label="Cmd Lin X", linestyle="--")
    axs[0, 1].plot(steps, obs[:, 10], label="Cmd Lin Y", linestyle="--")
    axs[0, 1].plot(steps, obs[:, 11], label="Cmd Ang Yaw", linestyle="--")
    axs[0, 1].set_title(f"Gravity & Velocity Commands (Robot {args.robot_idx})")
    axs[0, 1].set_ylabel("Value")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].grid(True)

    for i, j_name in enumerate(joint_names):
        axs[1, 0].plot(steps, obs[:, 12 + i], label=j_name)
    axs[1, 0].set_title(f"Joint Positions (Robot {args.robot_idx})")
    axs[1, 0].set_ylabel("Position (rad)")
    axs[1, 0].legend(loc="upper right", ncol=2, fontsize="small")
    axs[1, 0].grid(True)

    for i, j_name in enumerate(joint_names):
        axs[1, 1].plot(steps, obs[:, 24 + i], label=j_name)
    axs[1, 1].set_title(f"Joint Velocities (Robot {args.robot_idx})")
    axs[1, 1].set_ylabel("Velocity (rad/s)")
    axs[1, 1].legend(loc="upper right", ncol=2, fontsize="small")
    axs[1, 1].grid(True)

    for i, j_name in enumerate(joint_names):
        axs[2, 0].plot(steps, actions[:, i], label=j_name)
    axs[2, 0].set_title(f"Joint Actions (Robot {args.robot_idx})")
    axs[2, 0].set_xlabel("Time step")
    axs[2, 0].set_ylabel("Action (rad)")
    axs[2, 0].legend(loc="upper right", ncol=2, fontsize="small")
    axs[2, 0].grid(True)

    for i, j_name in enumerate(joint_names):
        axs[2, 1].plot(steps, obs[:, 36 + i], label=j_name)
    axs[2, 1].set_title(f"Previous Joint Actions (Robot {args.robot_idx})")
    axs[2, 1].set_xlabel("Time step")
    axs[2, 1].set_ylabel("Action (rad)")
    axs[2, 1].legend(loc="upper right", ncol=2, fontsize="small")
    axs[2, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=300)
    print(f"Plot saved to {args.save_path}")

if __name__ == "__main__":
    main()
