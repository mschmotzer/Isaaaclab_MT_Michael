# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch, math, numpy as np
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    # Check cube positions -> Use logical_and to combine conditions, every single criterion must be satisfied simultaneously
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)

    # Check gripper positions
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    return stacked

"""def eef_close_to_bottleneck_franka(
    env: ManagerBasedRLEnv,
    eef_name: str = "panda_hand",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Bottleneck pose attribute on env (set during backward augmentation)
    target_attr: str = "_bn_pose44",
    # Bottleneck gripper value attribute on env
    gripper_attr: str = "_bn_gripper_val",
    pos_tol: float = 0.1,            # 5 mm position tolerance
    ori_tol_deg: Optional[float] = None,  # orientation tolerance in degrees (None = ignore)
    use_gripper: bool = True,         # True to also check gripper state
    finger_joint_indices: Tuple[int, int] = (-2, -1),  # last two joints = fingers
    grip_tol: float = 0.005,          # 5 mm tolerance per finger
    # Cube working range parameters - matching the clipped ranges
    check_cube_range: bool = True,    # Whether to check if cubes are in working range
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    workspace_x_min: float = 0.35,    # Minimum x position for cubes
    workspace_x_max: float = 0.7,     # Maximum x position for cubes
    workspace_y_min: float = -0.3,    # Minimum y position for cubes
    workspace_y_max: float = 0.3,     # Maximum y position for cubes
    workspace_z_min: float = 0.0,     # Minimum z height for cubes
    workspace_z_max: float = 1.0,     # Maximum z height for cubes
) -> torch.Tensor:
    
    Returns a boolean tensor of shape [N] indicating, for each parallel env,
    whether the EEF is close to the bottleneck state defined during backward augmentation.
    
    The bottleneck state includes:
    - EEF pose (position and optionally orientation)
    - Gripper state (if use_gripper=True)
    
    This function is designed to work with the backward augmentation in data_generator_juicer.py
    where the bottleneck pose and gripper state are stored on the environment.
    
    device = getattr(env, "device", torch.device("cpu"))

    # --- Current EEF pose(s) -------------------------------------------------
    T_cur = env.get_robot_eef_pose(env_ids=None, eef_name=eef_name)  # [N,4,4] or [4,4]
    if not torch.is_tensor(T_cur):
        T_cur = torch.as_tensor(T_cur)
    T_cur = T_cur.to(device)
    if T_cur.ndim == 2:  # [4,4] -> broadcast to [N,4,4]
        N = int(getattr(env, "num_envs", 1))
        T_cur = T_cur.unsqueeze(0).expand(N, -1, -1).contiguous()
    elif T_cur.ndim == 3:
        N = T_cur.shape[0]
    else:
        raise ValueError(f"eef pose has unexpected shape {T_cur.shape} (expected [N,4,4] or [4,4])")

    # --- Target bottleneck pose(s) -------------------------------------------
    bn_pose = getattr(env, target_attr, None)
    if bn_pose is None:
        # No bottleneck target set -> no env is "close"
        #print(f"[BA] No bottleneck pose found at env.{target_attr}")
        return torch.zeros(N, dtype=torch.bool, device=device)

    if not torch.is_tensor(bn_pose):
        bn_pose = torch.as_tensor(bn_pose)
    bn_pose = bn_pose.to(device, dtype=T_cur.dtype)

    # Handle different shapes of bottleneck pose
    if bn_pose.ndim == 2:  # [4,4] single pose for all envs
        bn_pose = bn_pose.unsqueeze(0).expand(N, -1, -1).contiguous()
    elif bn_pose.ndim == 3 and bn_pose.shape[0] == 1:  # [1,4,4] -> broadcast to [N,4,4]
        bn_pose = bn_pose.expand(N, -1, -1).contiguous()
    elif bn_pose.ndim == 3 and bn_pose.shape[0] != N:
        #print(f"[BA] Warning: bottleneck pose shape {bn_pose.shape} doesn't match num_envs {N}")
        # Use the first pose for all envs
        bn_pose = bn_pose[0:1].expand(N, -1, -1).contiguous()

    # --- Position test -------------------------------------------------------
    p_cur = T_cur[:, :3, 3]  # [N, 3]
    p_tgt = bn_pose[:, :3, 3]  # [N, 3]
    pos_dist = torch.linalg.norm(p_cur - p_tgt, dim=1)  # [N]
    pos_ok = pos_dist <= pos_tol

    #print(f"[BA] Position check: distances={pos_dist.cpu().numpy()}, threshold={pos_tol}, ok={pos_ok.cpu().numpy()}")

    # --- Orientation test (optional) -----------------------------------------
    if ori_tol_deg is None:
        ori_ok = torch.ones(N, dtype=torch.bool, device=device)
    else:
        R_cur = T_cur[:, :3, :3]  # [N, 3, 3]
        R_tgt = bn_pose[:, :3, :3]  # [N, 3, 3]
        # Compute relative rotation: R_rel = R_cur^T * R_tgt
        dR = torch.matmul(R_cur.transpose(1, 2), R_tgt)  # [N, 3, 3]
        # Extract angle from rotation matrix using trace
        tr = dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]  # [N]
        # Numerical safety: clamp to valid range for arccos
        cos_arg = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
        ang_rad = torch.arccos(cos_arg)  # [N]
        ang_deg = ang_rad * (180.0 / math.pi)  # [N]
        ori_ok = ang_deg <= ori_tol_deg

        #print(f"[BA] Orientation check: angles={ang_deg.cpu().numpy()}, threshold={ori_tol_deg}, ok={ori_ok.cpu().numpy()}")

    # --- Gripper test (optional) ---------------------------------------------
    #if use_gripper:
    robot: Articulation = env.scene[robot_cfg.name]
    jpos = robot.data.joint_pos  # [N, dof]
    
    # Get target gripper value from environment
    bn_gripper = getattr(env, gripper_attr, None)
    if bn_gripper is None:
        #print(f"[BA] No bottleneck gripper value found at env.{gripper_attr}, skipping gripper check")
        grip_ok = torch.ones(N, dtype=torch.bool, device=device)
    else:
        if not torch.is_tensor(bn_gripper):
            bn_gripper = torch.as_tensor(bn_gripper, dtype=jpos.dtype, device=device)
        else:
            bn_gripper = bn_gripper.to(dtype=jpos.dtype, device=device)
        
        # Handle scalar or tensor gripper values
        if bn_gripper.ndim == 0:  # scalar
            target_grip = bn_gripper.item()
        else:
            target_grip = bn_gripper[0].item() if len(bn_gripper) > 0 else 0.04
        
        i1, i2 = finger_joint_indices
        g1 = torch.isclose(jpos[:, i1], torch.tensor(target_grip, dtype=jpos.dtype, device=device), atol=grip_tol, rtol=0.0)
        g2 = torch.isclose(jpos[:, i2], torch.tensor(target_grip, dtype=jpos.dtype, device=device), atol=grip_tol, rtol=0.0)
        grip_ok = g1 & g2
            
            #print(f"[BA] Gripper check: current=[{jpos[0, i1].item():.4f}, {jpos[0, i2].item():.4f}], target={target_grip:.4f}, ok={grip_ok.cpu().numpy()}")
    #else:
       # grip_ok = torch.ones(N, dtype=torch.bool, device=device)

    if check_cube_range:
        try:
            cube_1: RigidObject = env.scene[cube_1_cfg.name]
            cube_2: RigidObject = env.scene[cube_2_cfg.name]
            cube_3: RigidObject = env.scene[cube_3_cfg.name]
            
            # Get cube positions
            cube_1_pos = cube_1.data.root_pos_w  # [N, 3]
            cube_2_pos = cube_2.data.root_pos_w  # [N, 3]
            cube_3_pos = cube_3.data.root_pos_w  # [N, 3]
            
            # Check if all cubes are within the specified workspace bounds
            # X bounds: 0.35 to 0.7
            cubes_in_x_range = (cube_1_pos[:, 0] >= workspace_x_min) & (cube_1_pos[:, 0] <= workspace_x_max) & \
                              (cube_2_pos[:, 0] >= workspace_x_min) & (cube_2_pos[:, 0] <= workspace_x_max) & \
                              (cube_3_pos[:, 0] >= workspace_x_min) & (cube_3_pos[:, 0] <= workspace_x_max)
            
            # Y bounds: -0.3 to 0.3
            cubes_in_y_range = (cube_1_pos[:, 1] >= workspace_y_min) & (cube_1_pos[:, 1] <= workspace_y_max) & \
                              (cube_2_pos[:, 1] >= workspace_y_min) & (cube_2_pos[:, 1] <= workspace_y_max) & \
                              (cube_3_pos[:, 1] >= workspace_y_min) & (cube_3_pos[:, 1] <= workspace_y_max)
            
            # Z bounds: workspace_z_min to workspace_z_max
            cubes_in_z_range = (cube_1_pos[:, 2] >= workspace_z_min) & (cube_1_pos[:, 2] <= workspace_z_max) & \
                              (cube_2_pos[:, 2] >= workspace_z_min) & (cube_2_pos[:, 2] <= workspace_z_max) & \
                              (cube_3_pos[:, 2] >= workspace_z_min) & (cube_3_pos[:, 2] <= workspace_z_max)
            
            cube_range_ok = cubes_in_x_range & cubes_in_y_range & cubes_in_z_range
            
            #print(f"[BA] Cube range check: cube1=[{cube_1_pos[0, 0].item():.3f}, {cube_1_pos[0, 1].item():.3f}, {cube_1_pos[0, 2].item():.3f}], "
            #      f"x_range=[{workspace_x_min}, {workspace_x_max}], y_range=[{workspace_y_min}, {workspace_y_max}], "
            #      f"z_range=[{workspace_z_min}, {workspace_z_max}], ok={cube_range_ok.cpu().numpy()}")
            
        except (KeyError, AttributeError) as e:
            #print(f"[BA] Warning: Could not check cube range due to missing cubes: {e}")
            cube_range_ok = torch.ones(N, dtype=torch.bool, device=device)
    else:
        cube_range_ok = torch.ones(N, dtype=torch.bool, device=device)

    result =  grip_ok
    return result
    #print(f"[BA] Final bottleneck check result: {result.cpu().numpy()}")& ori_ok pos_ok & cube_range_ok &"""

def eef_close_to_bottleneck_franka(
    env: ManagerBasedRLEnv,
    eef_name: str = "panda_hand",  # Add this parameter
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    target_attr: str = "_cube_z_pos",
    pos_tol: float = 0.01,
    **kwargs,  # Accept any additional parameters
):
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    
    cube_1_z = cube_1.data.root_pos_w[:, 2]  # [N]
    cube_2_z = cube_2.data.root_pos_w[:, 2]  # [N]
    cube_3_z = cube_3.data.root_pos_w[:, 2]  # [N]
    
    cube_z_pos = getattr(env, target_attr, None)
    #print(f"Cube GoalPos Attribute: {target_attr} Value: {cube_z_pos} Type: {type(cube_z_pos)}")
    
    if cube_z_pos is None:
        # Handle case where cube_z_pos is not set
        cube_z_pos = torch.tensor([0.0, 0.0, 0.0], device=cube_1_z.device, dtype=cube_1_z.dtype)

    #print(f"Cube GoalPos: {cube_z_pos} Cube 1: {cube_1_z} Cube 2: {cube_2_z} Cube 3: {cube_3_z}")
    
    # Check cube positions -> Use logical_and to combine conditions
    stacked = torch.logical_and(
        torch.abs(cube_1_z - cube_z_pos[0]) < pos_tol, 
        torch.abs(cube_2_z - cube_z_pos[1]) < pos_tol
    )
    stacked = torch.logical_and(
        torch.abs(cube_3_z - cube_z_pos[2]) < pos_tol, 
        stacked
    )
    
    return stacked