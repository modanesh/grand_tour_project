import numpy as np
import zarr
import matplotlib.pyplot as plt
import argparse
import json


class GrandTourDataloader:
    def __init__(self, frequency: int = 10, mission_name_short: str = "ETH-1"):
        self.frequency = frequency  # number of samples per second
        self.mission_name_short = mission_name_short

        # load offset start and end unix absolute timestamps from json
        with open("data/config.json", "r") as f:
            offset_start_end_unix_absolute = json.load(f)
            self.offset_start_unix_absolute = offset_start_end_unix_absolute[self.mission_name_short]["offset_start_unix_absolute"]
            self.offset_end_unix_absolute = offset_start_end_unix_absolute[self.mission_name_short]["offset_end_unix_absolute"]

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

        # grand tour closest timestamp 
        self.absolute_timestamps = []
        self.adj_base_lin_vel = []
        self.adj_base_ang_vel = []
        self.adj_projected_gravity = []
        self.adj_velocity_commands = []
        self.adj_joint_pos = dict()
        self.adj_joint_vel = dict()

        self.load_data(self.mission_name_short)

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

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def load_data(self, mission_name_short):
        # load state odometry timestamps
        z = zarr.open(f"./data/{mission_name_short}/anymal_state_odometry/timestamp", mode="r")
        print(f"anymal_state_odometry timestamps shape:", z.shape)  # shape is (nrows)
        self.odometry_timestamps = z[:]
        # load base lin vel from state odometry
        z = zarr.open(f"./data/{mission_name_short}/anymal_state_odometry/twist_lin", mode="r")
        print(f"base_lin_vel shape:", z.shape)  # shape is (nrows)
        self.raw_base_lin_vel = z[:]
        # load base ang vel from state odometry
        z = zarr.open(f"./data/{mission_name_short}/anymal_state_odometry/twist_ang", mode="r")
        print(f"base_ang_vel shape:", z.shape)  # shape is (nrows)
        self.raw_base_ang_vel = z[:]
        # # load projected gravity from state odometry
        # gravity vector from pose_orientation
        z = zarr.open(f"./data/{mission_name_short}/anymal_state_odometry/pose_orien", mode="r")
        print(f"pose_orientation shape:", z.shape)  # shape is (nrows, 4)
        self.raw_pose_orientation = z[:]
        # convert quaternion to rotation matrix and get gravity vector
        self.raw_projected_gravity = np.array(
            [self.quaternion_to_rotation_matrix(q)[:3, 2] for q in self.raw_pose_orientation]
        )
        # z = zarr.open(f"./data/anymal_state_odometry/projected_gravity", mode="r")
        # print(f"projected_gravity shape:", z.shape) # shape is (nrows, 3)
        # self.odometry_projected_gravity = z[:]

        # load timestamps from command twist
        z = zarr.open(f"./data/{mission_name_short}/anymal_command_twist/timestamp", mode="r")
        print(f"anymal_command_twist timestamps shape:", z.shape)  # shape is (nrows)
        self.command_timestamps = z[:]
        # load linear velocity commands from command twist
        z = zarr.open(f"./data/{mission_name_short}/anymal_command_twist/linear", mode="r")
        print(f"anymal_command_twist linear shape:", z.shape)  # shape is (nrows, 3)
        self.raw_velocity_commands = z[:]


        # load timestamps from state actuator
        z = zarr.open(f"./data/{mission_name_short}/anymal_state_actuator/timestamp", mode="r")
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
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_state_joint_position", mode="r"
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
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_state_joint_velocity", mode="r"
            )
            self.raw_joint_vel[joint_name] = z[:]
            print(
                f"{joint_name} {key_idx}: joint_velocity shape:", z.shape
            )  # shape is (nrows)
            grand_tour_dict_joint_velocities[joint_name] = z[:]
            print(
                f"sample joint velocity: {grand_tour_dict_joint_velocities[joint_name][10000]}"
            )

        # process data based on frequency and closest timestamps
        for i in range(
            int(self.frequency * (self.offset_end_unix_absolute - self.offset_start_unix_absolute))
        ):
            timestamp = self.offset_start_unix_absolute + i / self.frequency
            if i % 100 == 0:
                print(f"timestamp: {timestamp}")

            # find closest timestamp for odometry
            closest_idx = self.get_closest_value_from_timestamp(timestamp, self.odometry_timestamps)
            # print(f"closest_idx: {closest_idx}")
            self.absolute_timestamps.append(timestamp)
            self.adj_base_lin_vel.append(self.raw_base_lin_vel[closest_idx])
            self.adj_base_ang_vel.append(self.raw_base_ang_vel[closest_idx])
            self.adj_projected_gravity.append(self.raw_projected_gravity[closest_idx])

            # find closest timestamp for command twist
            closest_idx = self.get_closest_value_from_timestamp(timestamp, self.command_timestamps)
            # print(f"closest_idx: {closest_idx}")
            self.adj_velocity_commands.append(self.raw_velocity_commands[closest_idx])

            # find closest timestamp for joint positions and velocities
            for joint_name in grand_tour_ref_keys_order:
                if joint_name not in self.adj_joint_pos:
                    self.adj_joint_pos[joint_name] = []
                    self.adj_joint_vel[joint_name] = []
                closest_idx = self.get_closest_value_from_timestamp(timestamp, self.actuator_timestamps)
                self.adj_joint_pos[joint_name].append(self.raw_joint_pos[joint_name][closest_idx])
                self.adj_joint_vel[joint_name].append(self.raw_joint_vel[joint_name][closest_idx])

        # convert to numpy arrays
        self.adj_base_lin_vel = np.array(self.adj_base_lin_vel)
        self.adj_base_ang_vel = np.array(self.adj_base_ang_vel)
        self.adj_projected_gravity = np.array(self.adj_projected_gravity)
        self.adj_velocity_commands = np.array(self.adj_velocity_commands)
        for joint_name in grand_tour_ref_keys_order:
            self.adj_joint_pos[joint_name] = np.array(self.adj_joint_pos[joint_name])
            self.adj_joint_vel[joint_name] = np.array(self.adj_joint_vel[joint_name])

        print("adj_base_lin_vel shape:", self.adj_base_lin_vel.shape)
        print("adj_base_ang_vel shape:", self.adj_base_ang_vel.shape)
        print("adj_projected_gravity shape:", self.adj_projected_gravity.shape)
        print("adj_velocity_commands shape:", self.adj_velocity_commands.shape)
        print("adj_joint_pos keys:", list(self.adj_joint_pos.keys()))
        print("adj_joint_vel keys:", list(self.adj_joint_vel.keys()))


    def get_observations_isaac_lab_format(self):
        """
        #[INFO] Observation Manager: <ObservationManager> contains 1 groups.
        +---------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (48,)) |
        +-----------+---------------------------------+-----------+
        |   Index   | Name                            |   Shape   |
        +-----------+---------------------------------+-----------+
        |     0     | base_lin_vel                    |    (3,)   |
        |     1     | base_ang_vel                    |    (3,)   |
        |     2     | projected_gravity               |    (3,)   |
        |     3     | velocity_commands               |    (3,)   |
        |     4     | joint_pos                       |   (12,)   |
        |     5     | joint_vel                       |   (12,)   |
        |     6     | actions                         |   (12,)   |
        +-----------+---------------------------------+-----------+
        """

        # Collect joint positions and velocities in order
        joint_pos_array = np.column_stack([self.adj_joint_pos[joint_name] for joint_name in self.isaac_lab_ref_keys_order])
        joint_vel_array = np.column_stack([self.adj_joint_vel[joint_name] for joint_name in self.isaac_lab_ref_keys_order])
        
        # Concatenate all observations along axis 1 to get shape (500, 36)
        observations = np.concatenate([
            self.adj_base_lin_vel,      # (500, 3)
            self.adj_base_ang_vel,      # (500, 3)
            self.adj_projected_gravity, # (500, 3)
            self.adj_velocity_commands, # (500, 3)
            joint_pos_array,            # (500, 12)
            joint_vel_array             # (500, 12)
        ], axis=1)
        
        # Return observations from idx 0 to n-1 (exclude last timestep)
        return observations[:-1]  # shape: (499, 36)

    def get_actions_isaac_lab_format(self):
        # Actions are typically joint positions or commands
        # Return joint positions shifted by one timestep (idx 1 to n)
        joint_pos_array = np.column_stack([self.adj_joint_pos[joint_name] for joint_name in self.isaac_lab_ref_keys_order])
        return joint_pos_array[1:]  # shape: (499, 12) - shifted by one timestep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mission_name_short", type=str, default="ETH-1")
    args = parser.parse_args()
    
    dataloader = GrandTourDataloader(mission_name_short=args.mission_name_short)
    plt.plot(
        dataloader.command_timestamps,
        dataloader.raw_velocity_commands[:, 0],
    )
    plt.plot(
        dataloader.actuator_timestamps,
        dataloader.raw_joint_pos["LF_HAA"],
    )
    print(f"baseline: {dataloader.command_timestamps[0]}")
    print(f"baseline + 100 sec: {dataloader.command_timestamps[0] + 100}")
    print(f"baseline + 150 sec: {dataloader.command_timestamps[0] + 150}")

    # plt.plot([dataloader.command_timestamps[0] + 100, dataloader.command_timestamps[0] + 100], [0, 1]
    # plt.plot([dataloader.command_timestamps[0] + 150, dataloader.command_timestamps[0] + 150], [0, 1], 'r--'), 'r--')


    plt.plot(dataloader.absolute_timestamps, dataloader.adj_velocity_commands[:, 0], label="velocity_command x")
    plt.plot(dataloader.absolute_timestamps, dataloader.adj_velocity_commands[:, 1], label="velocity_command y")
    plt.plot(dataloader.absolute_timestamps, dataloader.adj_velocity_commands[:, 2], label="velocity_command z")


    plt.plot(dataloader.absolute_timestamps, dataloader.adj_joint_pos["LF_HAA"], label="LF_HAA")
    plt.legend()
    plt.show()
    plt.savefig("velocity_command.png")

    plt.close()

    obs = dataloader.get_observations_isaac_lab_format()
    print("observations shape:", obs.shape)
    act = dataloader.get_actions_isaac_lab_format()
    print("actions shape:", act.shape)
    plt.plot(obs[:, 12])
    plt.plot(act[:, 0])
    # plt.show()
    plt.savefig("obs_act.png")
    plt.close()
