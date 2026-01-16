# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from . import mdp

"""This file defines the foundational configuration for the stack environment.
It includes the scene configuration, MDP settings, and event configurations."""


##
# Scene definition -> defines the scene with a table, robot "Franka Emika", and objects
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot-configuration and end-effector frames
    """

    # robots: will be populated by the derived env cfg
    robot: ArticulationCfg = MISSING # type: ignore
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING # type: ignore

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        # Add table offset in z direction to match the real robots configuration -> Offset coordinate frame by 0.011m
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.011], rot=[0.707, 0, 0, 0.707]), # type: ignore
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1, 1, 0.6762),  # (x_scale, y_scale, z_scale)
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.68]), # type: ignore
        spawn=GroundPlaneCfg(), # type: ignore
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg -> Set to Abs IK control for our stacking task
    arm_action: mdp.JointPositionActionCfg = MISSING # type: ignore
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING # type: ignore


# Define three observation groups: Policy, RGBCameraPolicy, and SubtaskCfg
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # Implement observation noise
        object = ObsTerm(func=mdp.object_obs_with_noise,
                        params={"position_noise_std": 0.002, "orientation_noise_std": 0.01})
        eef_pos = ObsTerm(func=mdp.ee_frame_pos_with_noise,
                        params={"noise_std": 0.002})
        eef_quat = ObsTerm(func=mdp.ee_frame_quat_with_noise,
                        params={"noise_std": 0.002})
        gripper_pos = ObsTerm(func=mdp.gripper_pos_with_noise,
                        params={"noise_std": 0.002})

        # Keep other observations as they are
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
    
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # cube_1: blue, cube_2: red, cube_3: green
        grasp_1 = ObsTerm(
            func=mdp.object_grasped, # Checks if cube_1 is grasped by the robot
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked, # Checks if cube_2 is stacked on cube_1
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped, # Checks if cube_3 is grasped by the robot
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Create observation groups "key value pairs"
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

# The domain randomization events are applied during environment resets
@configclass 
class DomainRandomizationCfg:
    """Domain randomization settings for the stacking environment."""

    # Use IsaacLab's built-in actuator gains randomization
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # Built-in function
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.50, 1.5),  # ±50% variation
            "damping_distribution_params": (0.50, 1.5),  # ±50% variation
            "operation": "scale",  # Scale the base values
            "distribution": "uniform",
        },
    )

    # Use IsaacLab's built-in mass randomization
    randomize_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # Built-in function
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.95, 1.05),  # ±5% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Randomize cube masses using built-in function
    randomize_cube_1_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_1"),
            "mass_distribution_params": (0.95, 1.05),  # ±5% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_cube_2_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_2"),
            "mass_distribution_params": (0.95, 1.05),  # ±5% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_cube_3_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_3"),
            "mass_distribution_params": (0.95, 1.05),  # ±5% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_control_latency = EventTerm(
        func=franka_stack_events.randomize_control_latency,  # Custom function for control latency
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "latency_steps_range": (0, 3),  # 0-3 timesteps delay (0-150ms at 20Hz)
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )
    # Success condition: all cubes stacked -> Very important for annotate_demos.py
    success = DoneTerm(func=mdp.cubes_stacked)

# Define the main stacking environment configuration -> Inherits from ManagerBasedRLEnvCfg "Manager based workflow"
@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    # Domain randomization settings
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # Unused managers -> These are not used in imitation learning pipelines
    commands = None
    rewards = None
    events = None
    curriculum = None

    # Configuration for viewing and interacting with the environment through an XR device.
    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # Control frequency is 20Hz (50ms), decimation is 5, so simulation runs at 100Hz (10ms)
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        # Enable domain randomization by default (can be disabled in child configs)
        self.enable_domain_randomization = False   

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
