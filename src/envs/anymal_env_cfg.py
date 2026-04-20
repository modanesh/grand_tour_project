"""Environment configuration for ANYMAL robot with camera support."""

from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.actuators import (
    ImplicitActuatorCfg,
    IdealPDActuatorCfg,
)
from scipy.spatial.transform import Rotation

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg,
)


class AnymalDFlatCameraEnvCfg:
    """Configuration for ANYMAL D robot with camera support in flat environment."""

    def __init__(
        self,
        env_type="flat",
        enable_cameras=False,
        actuation_mode="SEA",
        action_scale=1.0,
        action_offset=0.0,
        actuator_stiffness=100.0,
        actuator_damping=6.0,
    ):
        """Initialize environment configuration.

        Args:
            env_type: Environment type ("flat" or "warehouse")
            enable_cameras: Whether to enable robot-mounted front camera
            actuation_mode: Actuator mode ("SEA", "PD", or "Implicit")
            action_scale: Scale for joint position actions
            action_offset: Offset for joint position actions
            actuator_stiffness: Stiffness for PD/Implicit actuators
            actuator_damping: Damping for PD/Implicit actuators
        """
        self.env_type = env_type
        self.enable_cameras = enable_cameras
        self.actuation_mode = actuation_mode
        self.action_scale = action_scale
        self.action_offset = action_offset
        self.actuator_stiffness = actuator_stiffness
        self.actuator_damping = actuator_damping

        # Initialize base configuration
        self.base_cfg = AnymalDFlatEnvCfg()

    def __post_init__(self):
        """Post-initialization configuration."""
        # Camera configuration based on environment type
        if self.env_type == "flat":
            # add tiled camera to the existing scene config
            self.base_cfg.scene.tiled_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(-12.0, 2.0, 3.0),
                    rot=(0.9945, 0.0, 0.1045, 0.0),
                    convention="world",
                ),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                width=640,
                height=480,
            )

            # optional: nicer viewer pose
            self.base_cfg.viewer.eye = (7.0, 0.0, 3.0)
            self.base_cfg.viewer.lookat = (0.0, 0.0, 0.8)
        else:
            # add main scene camera - fixed in world space above robot spawn
            top_quat = Rotation.from_euler(
                "xyz", [-90, 0, 0], degrees=True
            ).as_quat()  # [x,y,z,w] -> [w,x,y,z]
            self.base_cfg.scene.tiled_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/top_cam",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(
                        0.0,
                        0.0,
                        3.0,
                    ),  # Fixed above robot spawn (robot at ~1m height)
                    rot=(
                        top_quat[3],
                        top_quat[0],
                        top_quat[1],
                        top_quat[2],
                    ),  # [w,x,y,z] for isaaclab
                    convention="world",
                ),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 50.0),  # Extended range for warehouse
                ),
                width=640,
                height=480,
            )

        # add robot-mounted front camera only if cameras are enabled
        if self.enable_cameras:
            front_quat = Rotation.from_euler(
                "xyz", [0, 0, 0], degrees=True
            ).as_quat()  # Forward looking (+X)
            self.base_cfg.scene.front_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.35, 0.0, 0.15),  # Front of robot, slightly up
                    rot=(
                        front_quat[3],
                        front_quat[0],
                        front_quat[1],
                        front_quat[2],
                    ),  # [w,x,y,z] for isaaclab
                    convention="world",
                ),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 20.0),
                ),
                width=640,
                height=480,
            )

        # optional: nicer viewer pose
        self.base_cfg.viewer.eye = (6.0, -4.0, 4.0)
        self.base_cfg.viewer.lookat = (0.0, 0.0, 0.5)

        ########### OBSERVATION AND ACTION OVERRIDE #########
        self.base_cfg.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, scale=1.0
        )
        self.base_cfg.observations.policy.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=1.0
        )
        self.base_cfg.observations.policy.projected_gravity = ObsTerm(
            func=mdp.projected_gravity, scale=1.0
        )
        # TODO: add velocity command
        # self.base_cfg.observations.policy.velocity_command = ObsTerm(func=mdp.velocity_command, scale=1.0)
        self.base_cfg.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos, scale=1.0
        )
        self.base_cfg.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel, scale=1.0
        )

        # position control instead of torque/effort control
        # Using absolute positions (reference actions are absolute joint positions)
        # Match original config: scale=0.5, use_default_offset=True (center around default pose)
        self.base_cfg.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=self.action_scale,
            offset=self.action_offset,
            use_default_offset=False,  # Use default joint positions as center
        )

        # Set initial joint positions to 0 for all joints
        # NOTE: Commented out to use default standing pose - required for reference tracking
        # self.base_cfg.scene.robot.init_state.joint_pos = {
        #     ".*": 0.0,  # All joints start at 0 rad
        # }

        # if the original config had another action term like joint_effort,
        # remove it so only joint position actions remain
        if hasattr(self.base_cfg.actions, "joint_effort"):
            self.base_cfg.actions.joint_effort = None
        if hasattr(self.base_cfg.actions, "torques"):
            self.base_cfg.actions.torques = None

        # Configure actuators based on actuation mode
        if self.actuation_mode == "SEA":
            # Use default SEA (Series Elastic Actuator) - LSTM-based from ANYMAL_D_CFG
            print("Using SEA (LSTM) actuation mode")
            # Keep default actuators from ANYMAL_D_CFG (already set)
        elif self.actuation_mode == "PD":
            print("Using PD actuation mode")
            self.base_cfg.scene.robot.actuators = {
                "legs": IdealPDActuatorCfg(
                    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                    stiffness=self.actuator_stiffness,
                    damping=self.actuator_damping,
                    effort_limit=80.0,
                    velocity_limit=7.5,
                )
            }
        elif self.actuation_mode == "Implicit":
            print("Using Implicit actuation mode")
            self.base_cfg.scene.robot.actuators = {
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=self.actuator_stiffness,
                    damping=self.actuator_damping,
                    effort_limit=1000.0,
                )
            }

    def get_cfg(self):
        """Get the complete environment configuration."""
        self.__post_init__()
        return self.base_cfg
