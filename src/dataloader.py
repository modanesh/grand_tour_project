import numpy as np
import zarr
import matplotlib.pyplot as plt
import argparse
import json


class GrandTourDataloader:
    def __init__(
        self,
        frequency: int = 30,
        mission_name_short: str = None,
        mission_names: list = None,
    ):
        self.frequency = frequency  # number of samples per second
        self.mission_name_short = mission_name_short
        self.mission_names = mission_names  # List of mission names to load

        # load all mission configs from json
        with open("data/config.json", "r") as f:
            self.mission_configs = json.load(f)

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

        # Data storage for all missions
        self.missions_data = {}
        self.combined_data = None

        # Load data based on provided arguments
        if mission_names is not None:
            # Load specific list of missions
            if not isinstance(mission_names, (list, tuple)):
                raise ValueError(
                    "mission_names must be a list or tuple of mission names"
                )
            for mission_name in mission_names:
                if mission_name not in self.mission_configs:
                    raise ValueError(
                        f"Mission '{mission_name}' not found in config.json"
                    )
                self.load_single_mission(mission_name)
            self._combine_all_missions()
        elif mission_name_short:
            if mission_name_short not in self.mission_configs:
                raise ValueError(
                    f"Mission '{mission_name_short}' not found in config.json"
                )
            self.load_single_mission(mission_name_short)
        else:
            self.load_all_missions()

        # print debug for loaded missions
        if self.mission_name_short:
            self._print_mission_debug(self.mission_name_short)
        else:
            for mission in self.missions_data:
                print(f"\n=== Mission: {mission} ===")
                self._print_mission_debug(mission)

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
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def load_all_missions(self):
        """Load data for all missions in config.json"""
        for mission_name in self.mission_configs:
            print(f"Loading mission: {mission_name}")
            self.load_single_mission(mission_name)
        self._combine_all_missions()

    def load_single_mission(self, mission_name_short):
        """Load data for a single mission"""
        # Initialize data storage for this mission
        self.missions_data[mission_name_short] = {
            "offset_start_unix_absolute": self.mission_configs[mission_name_short][
                "offset_start_unix_absolute"
            ],
            "offset_end_unix_absolute": self.mission_configs[mission_name_short][
                "offset_end_unix_absolute"
            ],
            "raw_state_odometry_timestamps": None,
            "raw_command_timestamps": None,
            "raw_actuator_timestamps": None,
            "raw_base_lin_vel": None,
            "raw_base_ang_vel": None,
            "raw_projected_gravity": None,
            "raw_velocity_commands": None,
            "raw_joint_pos": dict(),
            "raw_joint_vel": dict(),
            "absolute_timestamps": [],
            "adj_base_lin_vel": [],
            "adj_base_ang_vel": [],
            "adj_projected_gravity": [],
            "adj_velocity_commands": [],
            "adj_joint_pos": dict(),
            "adj_joint_vel": dict(),
            "adj_command_position": dict(),
        }

        mission_data = self.missions_data[mission_name_short]

        # load state odometry timestamps
        z = zarr.open(
            f"./data/{mission_name_short}/anymal_state_odometry/timestamp", mode="r"
        )
        print(f"anymal_state_odometry timestamps shape:", z.shape)  # shape is (nrows)
        mission_data["odometry_timestamps"] = z[:]
        # load base lin vel from state odometry
        z = zarr.open(
            f"./data/{mission_name_short}/anymal_state_odometry/twist_lin", mode="r"
        )
        print(f"base_lin_vel shape:", z.shape)  # shape is (nrows)
        mission_data["raw_base_lin_vel"] = z[:]
        # load base ang vel from state odometry
        z = zarr.open(
            f"./data/{mission_name_short}/anymal_state_odometry/twist_ang", mode="r"
        )
        print(f"base_ang_vel shape:", z.shape)  # shape is (nrows)
        mission_data["raw_base_ang_vel"] = z[:]
        # # load projected gravity from state odometry
        # gravity vector from pose_orientation
        z = zarr.open(
            f"./data/{mission_name_short}/anymal_state_odometry/pose_orien", mode="r"
        )
        print(f"pose_orientation shape:", z.shape)  # shape is (nrows, 4)
        mission_data["raw_pose_orientation"] = z[:]
        # convert quaternion to rotation matrix and get gravity vector
        mission_data["raw_projected_gravity"] = np.array(
            [
                self.quaternion_to_rotation_matrix(q)[:3, 2]
                for q in mission_data["raw_pose_orientation"]
            ]
        )

        # load timestamps from command twist
        z = zarr.open(
            f"./data/{mission_name_short}/anymal_command_twist/timestamp", mode="r"
        )
        print(f"anymal_command_twist timestamps shape:", z.shape)  # shape is (nrows)
        mission_data["command_timestamps"] = z[:]
        print("---")
        print(f"min command_timestamps: {mission_data['command_timestamps'].min()}")
        print(f"max command_timestamps: {mission_data['command_timestamps'].max()}")
        print("---")
        # load linear commands from command twist
        z = zarr.open(
            f"./data/{mission_name_short}/anymal_command_twist/linear", mode="r"
        )
        print(f"anymal_command_twist linear shape:", z.shape)  # shape is (nrows, 3)

        # Copy to numpy array so we can modify it (zarr is read-only)
        velocity_commands = z[:]  # Copy to numpy array

        # the 3rd col is yaw from angular velocity commands
        z_angular = zarr.open(
            f"./data/{mission_name_short}/anymal_command_twist/angular", mode="r"
        )
        print(
            f"anymal_command_twist angular shape:", z_angular.shape
        )  # shape is (nrows, 3)

        velocity_commands[:, 2] = z_angular[:, 2]

        mission_data["raw_velocity_commands"] = velocity_commands

        # load timestamps from state actuator
        z = zarr.open(
            f"./data/{mission_name_short}/anymal_state_actuator/timestamp", mode="r"
        )
        print(f"anymal_state_actuator timestamps shape:", z.shape)  # shape is (nrows)
        mission_data["actuator_timestamps"] = z[:]

        # load command fields from state actuator (stored as raw state)
        mission_data["raw_command_mode"] = dict()
        mission_data["raw_command_position"] = dict()
        mission_data["raw_command_velocity"] = dict()
        mission_data["raw_command_joint_torque"] = dict()
        mission_data["raw_command_pid_gains_d"] = dict()
        mission_data["raw_command_pid_gains_i"] = dict()
        mission_data["raw_command_pid_gains_p"] = dict()

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

        keys_ = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
        for key_idx, joint_name in zip(keys_, grand_tour_ref_keys_order):
            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_state_joint_position",
                mode="r",
            )
            mission_data["raw_joint_pos"][joint_name] = z[:]
            print(f"{joint_name} {key_idx}: joint_position shape:", z.shape)

            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_state_joint_velocity",
                mode="r",
            )
            mission_data["raw_joint_vel"][joint_name] = z[:]
            print(f"{joint_name} {key_idx}: joint_velocity shape:", z.shape)

            # load command fields from state actuator
            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_command_mode",
                mode="r",
            )
            mission_data["raw_command_mode"][joint_name] = z[:]

            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_command_position",
                mode="r",
            )
            mission_data["raw_command_position"][joint_name] = z[:]

            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_command_velocity",
                mode="r",
            )
            mission_data["raw_command_velocity"][joint_name] = z[:]

            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_command_joint_torque",
                mode="r",
            )
            mission_data["raw_command_joint_torque"][joint_name] = z[:]

            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_command_pid_gains_d",
                mode="r",
            )
            mission_data["raw_command_pid_gains_d"][joint_name] = z[:]

            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_command_pid_gains_i",
                mode="r",
            )
            mission_data["raw_command_pid_gains_i"][joint_name] = z[:]

            z = zarr.open(
                f"./data/{mission_name_short}/anymal_state_actuator/{key_idx}_command_pid_gains_p",
                mode="r",
            )
            mission_data["raw_command_pid_gains_p"][joint_name] = z[:]

        # process data based on frequency and closest timestamps
        for i in range(
            int(
                self.frequency
                * (
                    mission_data["offset_end_unix_absolute"]
                    - mission_data["offset_start_unix_absolute"]
                )
            )
        ):
            timestamp = mission_data["offset_start_unix_absolute"] + i / self.frequency
            if i % 10000 == 0:
                print(f"timestamp: {timestamp}")

            # find closest timestamp for odometry
            closest_idx = self.get_closest_value_from_timestamp(
                timestamp, mission_data["odometry_timestamps"]
            )
            mission_data["absolute_timestamps"].append(timestamp)
            mission_data["adj_base_lin_vel"].append(
                mission_data["raw_base_lin_vel"][closest_idx]
            )
            mission_data["adj_base_ang_vel"].append(
                mission_data["raw_base_ang_vel"][closest_idx]
            )
            mission_data["adj_projected_gravity"].append(
                mission_data["raw_projected_gravity"][closest_idx]
            )

            # find closest timestamp for command twist
            closest_idx = self.get_closest_value_from_timestamp(
                timestamp, mission_data["command_timestamps"]
            )
            mission_data["adj_velocity_commands"].append(
                mission_data["raw_velocity_commands"][closest_idx]
            )

            # find closest timestamp for joint positions and velocities
            for joint_name in grand_tour_ref_keys_order:
                if joint_name not in mission_data["adj_joint_pos"]:
                    mission_data["adj_joint_pos"][joint_name] = []
                    mission_data["adj_joint_vel"][joint_name] = []
                    mission_data["adj_command_position"][joint_name] = []
                closest_idx = self.get_closest_value_from_timestamp(
                    timestamp, mission_data["actuator_timestamps"]
                )
                mission_data["adj_joint_pos"][joint_name].append(
                    mission_data["raw_joint_pos"][joint_name][closest_idx]
                )
                mission_data["adj_joint_vel"][joint_name].append(
                    mission_data["raw_joint_vel"][joint_name][closest_idx]
                )
                mission_data["adj_command_position"][joint_name].append(
                    mission_data["raw_command_position"][joint_name][closest_idx]
                )

        # convert to numpy arrays
        mission_data["adj_base_lin_vel"] = np.array(mission_data["adj_base_lin_vel"])
        mission_data["adj_base_ang_vel"] = np.array(mission_data["adj_base_ang_vel"])
        mission_data["adj_projected_gravity"] = np.array(
            mission_data["adj_projected_gravity"]
        )
        mission_data["adj_velocity_commands"] = np.array(
            mission_data["adj_velocity_commands"]
        )
        for joint_name in grand_tour_ref_keys_order:
            mission_data["adj_joint_pos"][joint_name] = np.array(
                mission_data["adj_joint_pos"][joint_name]
            )
            mission_data["adj_joint_vel"][joint_name] = np.array(
                mission_data["adj_joint_vel"][joint_name]
            )
            mission_data["adj_command_position"][joint_name] = np.array(
                mission_data["adj_command_position"][joint_name]
            )

        print("adj_base_lin_vel shape:", mission_data["adj_base_lin_vel"].shape)
        print("adj_base_ang_vel shape:", mission_data["adj_base_ang_vel"].shape)
        print(
            "adj_projected_gravity shape:", mission_data["adj_projected_gravity"].shape
        )
        print(
            "adj_velocity_commands shape:", mission_data["adj_velocity_commands"].shape
        )
        print("adj_joint_pos keys:", list(mission_data["adj_joint_pos"].keys()))
        print("adj_joint_vel keys:", list(mission_data["adj_joint_vel"].keys()))

    def _combine_all_missions(self):
        """Combine data from all missions into a single dataset with mission boundaries"""
        print("\nCombining all missions...")

        combined_absolute_timestamps = []
        combined_adj_base_lin_vel = []
        combined_adj_base_ang_vel = []
        combined_adj_projected_gravity = []
        combined_adj_velocity_commands = []
        combined_adj_joint_pos = {joint: [] for joint in self.isaac_lab_ref_keys_order}
        combined_adj_joint_vel = {joint: [] for joint in self.isaac_lab_ref_keys_order}
        combined_adj_command_position = {joint: [] for joint in self.isaac_lab_ref_keys_order}

        # Store mission boundaries for proper observation/action handling
        mission_boundaries = []
        current_start = 0

        for mission_name, mission_data in self.missions_data.items():
            mission_length = len(mission_data["absolute_timestamps"])
            print(f"Adding mission {mission_name} with {mission_length} samples")

            combined_absolute_timestamps.extend(mission_data["absolute_timestamps"])
            combined_adj_base_lin_vel.extend(mission_data["adj_base_lin_vel"])
            combined_adj_base_ang_vel.extend(mission_data["adj_base_ang_vel"])
            combined_adj_projected_gravity.extend(mission_data["adj_projected_gravity"])
            combined_adj_velocity_commands.extend(mission_data["adj_velocity_commands"])

            for joint_name in self.isaac_lab_ref_keys_order:
                combined_adj_joint_pos[joint_name].extend(
                    mission_data["adj_joint_pos"][joint_name]
                )
                combined_adj_joint_vel[joint_name].extend(
                    mission_data["adj_joint_vel"][joint_name]
                )
                combined_adj_command_position[joint_name].extend(
                    mission_data["adj_command_position"][joint_name]
                )

            # Record mission boundaries: [start_idx, end_idx] inclusive
            mission_boundaries.append(
                [current_start, current_start + mission_length - 1]
            )
            current_start += mission_length

        # Convert to numpy arrays
        self.combined_data = {
            "absolute_timestamps": np.array(combined_absolute_timestamps),
            "adj_base_lin_vel": np.array(combined_adj_base_lin_vel),
            "adj_base_ang_vel": np.array(combined_adj_base_ang_vel),
            "adj_projected_gravity": np.array(combined_adj_projected_gravity),
            "adj_velocity_commands": np.array(combined_adj_velocity_commands),
            "adj_joint_pos": {
                joint: np.array(combined_adj_joint_pos[joint])
                for joint in self.isaac_lab_ref_keys_order
            },
            "adj_joint_vel": {
                joint: np.array(combined_adj_joint_vel[joint])
                for joint in self.isaac_lab_ref_keys_order
            },
            "adj_command_position": {
                joint: np.array(combined_adj_command_position[joint])
                for joint in self.isaac_lab_ref_keys_order
            },
            "mission_boundaries": mission_boundaries,
        }

        print(f"Combined dataset: {len(combined_absolute_timestamps)} total samples")
        print(f"Mission boundaries: {mission_boundaries}")

    def _print_mission_debug(self, mission_name):
        """Print debug information for a specific mission"""
        if mission_name in self.missions_data:
            data = self.missions_data[mission_name]
            print(f"Mission {mission_name}:")
            print("  raw_base_lin_vel shape:", data["raw_base_lin_vel"].shape)
            print("  raw_base_ang_vel shape:", data["raw_base_ang_vel"].shape)
            print("  raw_projected_gravity shape:", data["raw_projected_gravity"].shape)
            print("  raw_velocity_commands shape:", data["raw_velocity_commands"].shape)
            print("  raw_joint_pos keys:", list(data["raw_joint_pos"].keys()))
            print("  raw_joint_vel keys:", list(data["raw_joint_vel"].keys()))

    def load_data(self, mission_name_short):
        """Legacy method - use load_single_mission instead"""
        self.load_single_mission(mission_name_short)

    def get_observations_isaac_lab_format(self, mission_name: str = None):
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

        # Determine which data to use
        if mission_name:
            if mission_name not in self.missions_data:
                raise ValueError(f"Mission '{mission_name}' not found")
            data = self.missions_data[mission_name]
            # Return observations from idx 0 to n-1 (exclude last timestep)
            joint_pos_array = np.column_stack(
                [
                    data["adj_joint_pos"][joint_name]
                    for joint_name in self.isaac_lab_ref_keys_order
                ]
            )
            joint_vel_array = np.column_stack(
                [
                    data["adj_joint_vel"][joint_name]
                    for joint_name in self.isaac_lab_ref_keys_order
                ]
            )
            observations = np.concatenate(
                [
                    data["adj_base_lin_vel"],
                    data["adj_base_ang_vel"],
                    data["adj_projected_gravity"],
                    data["adj_velocity_commands"],
                    joint_pos_array,
                    joint_vel_array,
                ],
                axis=1,
            )
            return observations[:-1]
        elif self.combined_data is not None:
            # For combined data, return observations respecting mission boundaries
            all_observations = []
            for start_idx, end_idx in self.combined_data["mission_boundaries"]:
                # Get data for this mission segment
                mission_slice = slice(start_idx, end_idx + 1)

                # Collect joint positions and velocities for this mission
                joint_pos_array = np.column_stack(
                    [
                        self.combined_data["adj_joint_pos"][joint_name][mission_slice]
                        for joint_name in self.isaac_lab_ref_keys_order
                    ]
                )
                joint_vel_array = np.column_stack(
                    [
                        self.combined_data["adj_joint_vel"][joint_name][mission_slice]
                        for joint_name in self.isaac_lab_ref_keys_order
                    ]
                )

                # Concatenate observations for this mission
                mission_observations = np.concatenate(
                    [
                        self.combined_data["adj_base_lin_vel"][mission_slice],
                        self.combined_data["adj_base_ang_vel"][mission_slice],
                        self.combined_data["adj_projected_gravity"][mission_slice],
                        self.combined_data["adj_velocity_commands"][mission_slice],
                        joint_pos_array,
                        joint_vel_array,
                    ],
                    axis=1,
                )

                # Return 0 to n-1 for this mission (exclude last timestep)
                all_observations.append(mission_observations[:-1])

            # Concatenate all mission observations
            return np.concatenate(all_observations, axis=0)
        elif self.mission_name_short:
            data = self.missions_data[self.mission_name_short]
            joint_pos_array = np.column_stack(
                [
                    data["adj_joint_pos"][joint_name]
                    for joint_name in self.isaac_lab_ref_keys_order
                ]
            )
            joint_vel_array = np.column_stack(
                [
                    data["adj_joint_vel"][joint_name]
                    for joint_name in self.isaac_lab_ref_keys_order
                ]
            )
            observations = np.concatenate(
                [
                    data["adj_base_lin_vel"],
                    data["adj_base_ang_vel"],
                    data["adj_projected_gravity"],
                    data["adj_velocity_commands"],
                    joint_pos_array,
                    joint_vel_array,
                ],
                axis=1,
            )
            return observations[:-1]
        else:
            raise ValueError("No data available. Load missions first.")

    def get_actions_isaac_lab_format(
        self, mission_name: str = None, shift_by_one: bool = True
    ):
        # Actions are typically joint positions or commands
        # Return joint positions, optionally shifted by one timestep
        #
        # Args:
        #   shift_by_one: If True (default), returns actions[1:] for training (predict next action)
        #                 If False, returns actions[0:] for reference tracking (replay exact actions)

        # Determine which data to use
        if mission_name:
            if mission_name not in self.missions_data:
                raise ValueError(f"Mission '{mission_name}' not found")
            data = self.missions_data[mission_name]
            cmd_pos_array = np.column_stack(
                [
                    data["adj_command_position"][joint_name]
                    for joint_name in self.isaac_lab_ref_keys_order
                ]
            )
            if shift_by_one:
                return cmd_pos_array[1:]  # shifted by one timestep for training
            else:
                return cmd_pos_array  # no shift for reference tracking
        elif self.combined_data is not None:
            # For combined data, return actions respecting mission boundaries
            all_actions = []
            for start_idx, end_idx in self.combined_data["mission_boundaries"]:
                # Get data for this mission segment
                mission_slice = slice(start_idx, end_idx + 1)

                # Collect command positions for this mission
                cmd_pos_array = np.column_stack(
                    [
                        self.combined_data["adj_command_position"][joint_name][mission_slice]
                        for joint_name in self.isaac_lab_ref_keys_order
                    ]
                )

                # Return 1 to n for this mission (exclude first timestep)
                all_actions.append(cmd_pos_array[1:])

            # Concatenate all mission actions
            return np.concatenate(all_actions, axis=0)
        elif self.mission_name_short:
            data = self.missions_data[self.mission_name_short]
            cmd_pos_array = np.column_stack(
                [
                    data["adj_command_position"][joint_name]
                    for joint_name in self.isaac_lab_ref_keys_order
                ]
            )
            return cmd_pos_array[1:]  # shifted by one timestep
        else:
            raise ValueError("No data available. Load missions first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mission_name_short",
        type=str,
        default=None,
        help="Specific mission to load (or None for all missions)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Load all missions and combine them"
    )
    args = parser.parse_args()

    # Determine loading mode
    if args.all or args.mission_name_short is None:
        print("Loading all missions...")
        dataloader = GrandTourDataloader()  # Load all missions
        data_source = "combined"
    else:
        print(f"Loading single mission: {args.mission_name_short}")
        dataloader = GrandTourDataloader(mission_name_short=args.mission_name_short)
        data_source = args.mission_name_short

    # Get data for plotting
    if data_source == "combined":
        timestamps = dataloader.combined_data["absolute_timestamps"]
        velocity_commands = dataloader.combined_data["adj_velocity_commands"]
        joint_pos = dataloader.combined_data["adj_joint_pos"]
        obs = dataloader.get_observations_isaac_lab_format()
        act = dataloader.get_actions_isaac_lab_format()
    else:
        mission_data = dataloader.missions_data[data_source]
        timestamps = mission_data["absolute_timestamps"]
        velocity_commands = mission_data["adj_velocity_commands"]
        joint_pos = mission_data["adj_joint_pos"]
        obs = dataloader.get_observations_isaac_lab_format(mission_name=data_source)
        act = dataloader.get_actions_isaac_lab_format(mission_name=data_source)

    print(f"Data source: {data_source}")
    print(f"Total timestamps: {len(timestamps)}")

    # Get offset from appropriate data source
    if data_source == "combined":
        # For combined data, use the first mission's offset as reference
        first_mission = list(dataloader.missions_data.keys())[0]
        offset_start = dataloader.missions_data[first_mission][
            "offset_start_unix_absolute"
        ]
        offset_end = dataloader.missions_data[first_mission]["offset_end_unix_absolute"]
    else:
        offset_start = dataloader.missions_data[data_source][
            "offset_start_unix_absolute"
        ]
        offset_end = dataloader.missions_data[data_source]["offset_end_unix_absolute"]

    # Plotting
    plt.figure(figsize=(12, 8))

    # Velocity commands
    plt.subplot(2, 1, 1)
    # lower bound offset - vertical line spanning the plot
    plt.axvline(
        x=offset_start, color="orange", linestyle="--", label="timestamp lower bound"
    )
    # upper bound offset - vertical line spanning the plot
    plt.axvline(
        x=offset_end, color="orange", linestyle="--", label="timestamp upper bound"
    )
    plt.plot(timestamps, velocity_commands[:, 0], label="velocity_command x")
    plt.plot(timestamps, velocity_commands[:, 1], label="velocity_command y")
    plt.plot(timestamps, velocity_commands[:, 2], label="velocity_command yaw")
    plt.plot(timestamps, joint_pos["LF_HAA"], label="LF_HAA")
    plt.legend()
    plt.title(f"Velocity Commands and Joint Position ({data_source})")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Observations and actions
    plt.subplot(2, 1, 2)
    plt.plot(obs[:, 12], label="Observation (joint pos)")
    plt.plot(act[:, 0], label="Action (commanded pos)")
    plt.legend()
    plt.title("Observations vs Actions")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.xlim(0, 1000)

    plt.tight_layout()
    plt.savefig("velocity_command.png")
    plt.close()

    print("observations shape:", obs.shape)
    print("actions shape:", act.shape)
