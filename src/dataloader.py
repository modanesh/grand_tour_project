import numpy as np
import zarr
import matplotlib.pyplot as plt


class GrandTourDataloader:
    def __init__(self, frequency: int = 10):
        self.frequency = frequency  # number of samples per second
        self.offset_start_unix_absolute = 0.0
        self.offset_end_unix_absolute = 0.0
        self.isaac_lab_ref_keys_order = [
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

        # grand tour raw data
        self.raw_state_odometry_timestamps = None
        self.raw_command_timestamps = None
        self.raw_actuator_timestamps = None

        self.raw_base_lin_vel = None
        self.raw_base_ang_vel = None
        self.raw_projected_gravity = None
        self.raw_velocity_commands = None
        self.raw_joint_pos = dict()
        self.raw_joint_vel = dict()

        self.load_data()

        # print debug
        print("raw_base_lin_vel shape:", self.raw_base_lin_vel.shape)
        print("raw_base_ang_vel shape:", self.raw_base_ang_vel.shape)
        print("raw_projected_gravity shape:", self.raw_projected_gravity.shape)
        print("raw_velocity_commands shape:", self.raw_velocity_commands.shape)
        print("raw_joint_pos keys:", list(self.raw_joint_pos.keys()))
        print("raw_joint_vel keys:", list(self.raw_joint_vel.keys()))

    def get_closest_value_from_timestamp(self, target_timestamp, timestamps):
        """Find the closest timestamp and return its index."""
        idx = np.searchsorted(timestamps, target_timestamp)
        if idx == 0:
            return 0
        elif idx == len(timestamps):
            return len(timestamps) - 1
        else:
            # Check which neighbor is closer
            left_diff = target_timestamp - timestamps[idx - 1]
            right_diff = timestamps[idx] - target_timestamp
            return idx - 1 if left_diff <= right_diff else idx

    def load_data(self):
        # load state odometry timestamps
        z = zarr.open(f"./data/anymal_state_odometry/timestamp", mode="r")
        print(f"anymal_state_odometry timestamps shape:", z.shape)  # shape is (nrows)
        self.odometry_timestamps = z[:]
        # load base lin vel from state odometry
        z = zarr.open(f"./data/anymal_state_odometry/twist_lin", mode="r")
        print(f"base_lin_vel shape:", z.shape)  # shape is (nrows)
        self.raw_base_lin_vel = z[:]
        # load base ang vel from state odometry
        z = zarr.open(f"./data/anymal_state_odometry/twist_ang", mode="r")
        print(f"base_ang_vel shape:", z.shape)  # shape is (nrows)
        self.raw_base_ang_vel = z[:]
        # # load projected gravity from state odometry
        # temporarily is just (0,0,1)
        self.raw_projected_gravity = np.array(
            [[0.0, 0.0, 1.0] for _ in range(len(self.odometry_timestamps))]
        )
        # z = zarr.open(f"./data/anymal_state_odometry/projected_gravity", mode="r")
        # print(f"projected_gravity shape:", z.shape) # shape is (nrows, 3)
        # self.odometry_projected_gravity = z[:]

        # load timestamps from command twist
        z = zarr.open(f"./data/anymal_command_twist/timestamp", mode="r")
        print(f"anymal_command_twist timestamps shape:", z.shape)  # shape is (nrows)
        self.command_timestamps = z[:]
        # load linear velocity commands from command twist
        z = zarr.open(f"./data/anymal_command_twist/linear", mode="r")
        print(f"anymal_command_twist linear shape:", z.shape)  # shape is (nrows, 3)
        self.raw_velocity_commands = z[:]

        # load timestamps from state actuator
        z = zarr.open(f"./data/anymal_state_actuator/timestamp", mode="r")
        print(f"anymal_state_actuator timestamps shape:", z.shape)  # shape is (nrows)
        self.actuator_timestamps = z[:]

        grand_tour_ref_keys_order = [
            "LF_HAA",
            "LF_HFE",
            "LF_KFE",
            "RF_HAA",
            "RF_HFE",
            "RF_KFE",
            "LH_HAA",
            "LH_HFE",
            "LH_KFE",
            "RH_HAA",
            "RH_HFE",
            "RH_KFE",
        ]
        grand_tour_dict_joint_positions = dict()
        grand_tour_dict_joint_velocities = dict()

        keys_ = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
        joint_positions_all = []
        for key_idx, joint_name in zip(keys_, grand_tour_ref_keys_order):
            z = zarr.open(
                f"./data/anymal_state_actuator/{key_idx}_state_joint_position", mode="r"
            )
            self.raw_joint_pos[joint_name] = z[:]
            print(
                f"{joint_name} {key_idx}: joint_position shape:", z.shape
            )  # shape is (nrows)
            grand_tour_dict_joint_positions[joint_name] = z[:]
            print(
                f"sample joint position: {grand_tour_dict_joint_positions[joint_name][10000]}"
            )

            z = zarr.open(
                f"./data/anymal_state_actuator/{key_idx}_state_joint_velocity", mode="r"
            )
            self.raw_joint_vel[joint_name] = z[:]
            print(
                f"{joint_name} {key_idx}: joint_velocity shape:", z.shape
            )  # shape is (nrows)
            grand_tour_dict_joint_velocities[joint_name] = z[:]
            print(
                f"sample joint velocity: {grand_tour_dict_joint_velocities[joint_name][10000]}"
            )

    def get_observations(self):
        pass

    def get_actions(self):
        pass


if __name__ == "__main__":
    dataloader = GrandTourDataloader()
    plt.plot(
        (dataloader.command_timestamps - dataloader.command_timestamps[0]) / 1e9,
        dataloader.raw_velocity_commands[: 0],
    )
    plt.savefig("velocity_command.png")
