


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
    save_observations: If provided, save successful observations to file.
    obs_output_file: Output file path for saved observations.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
from collections import deque
import pickle
import json 
import numpy as np
import pandas as pd  # Add pandas import for CSV handling
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor.")
parser.add_argument("--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor.")
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

# Arguments for observation logging
parser.add_argument("--save_observations", action="store_true", default=False, help="Save successful observations to file.")
parser.add_argument("--obs_output_file", type=str, default="successful_observations.csv", help="Output file path for saved observations.")
parser.add_argument("--csv_output_file", type=str, default="successful_observations.csv", help="Output CSV file path for detailed observations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device, save_observations=False, trial_id=0):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        success_term: The success termination condition.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.
        save_observations: Whether to save observations for logging.
        trial_id: The trial number for this rollout.

    Returns:
        terminated: Whether the rollout terminated successfully.
        traj: The trajectory of the rollout.
        observation_log: List of observations if save_observations is True, else None.
        csv_data: List of CSV-formatted observation data if save_observations is True, else None.
    """
    # Episode initialization -> Reset the internal states
    policy.start_episode()
    obs_dict, _ = env.reset()

    # Initialize trajectory storage
    traj = dict(actions=[], obs=[], next_obs=[])

    # Add observation logging 
    observation_log = [] if save_observations else None
    csv_data = [] if save_observations else None
    

    training_obs_keys = ["eef_pos", "eef_quat", "gripper_pos", "object","image"]
    ob_buffer=[]
    context_length = 1  # Default context length
    # Main rollout loop
    try:
            # Access the algo config from the policy
            algo_config = policy.policy.global_config.algo
            
            if hasattr(algo_config, 'transformer') and algo_config.transformer.enabled:
                context_length = algo_config.transformer.context_length
                seq_length = policy.policy.global_config.train.seq_length

            else:
                print("[INFO] Model is not using transformer architecture")
                
    except AttributeError as e:
            print(f"[WARNING] Could not read model configuration: {e}")
    obs = copy.deepcopy(obs_dict["policy"]) # Access the PolicyCfg observations
    for ob in obs:
        obs[ob] = torch.squeeze(obs[ob])

    # Check if environment image observations -> Not needed for stack environments
    if hasattr(env.cfg, "image_obs_list"):
        # Process image observations for robomimic inference
        for image_name in env.cfg.image_obs_list:
            if image_name in obs_dict["policy"].keys():
                # Convert from chw uint8 to hwc normalized float
                image = torch.squeeze(obs_dict["policy"][image_name])
                image = image.permute(2, 0, 1).clone().float()
                image = image / 255.0
                image = image.clip(0.0, 1.0)
                obs[image_name] = image

    # 2. Filter observations to match training configuration
    training_obs_keys = ["eef_pos", "eef_quat", "gripper_pos", "object"]
    filtered_obs = {key: obs[key] for key in training_obs_keys if key in obs}

    ob_buffer = deque(maxlen=context_length)

    # Then the buffer management becomes simply:
    # 3. Manage observation buffer with context_length
    while len(ob_buffer) < context_length:
        ob_buffer.append(filtered_obs)
        print(":) Filling observation buffer... Current size:", len(ob_buffer))

    for i in range(horizon):
        # 1. Observation Preprocessing
        obs = copy.deepcopy(obs_dict["policy"]) # Access the PolicyCfg observations
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # Check if environment image observations -> Not needed for stack environments
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        # 2. Filter observations to match training configuration
        training_obs_keys = ["eef_pos", "eef_quat", "gripper_pos", "object"]
        filtered_obs = {key: obs[key] for key in training_obs_keys if key in obs}

        

        # Then the buffer management becomes simply:
        # 3. Manage observation buffer with context_length
        ob_buffer.append(filtered_obs)
        import numpy as np

        ob_buffer.append(filtered_obs)
        # Convert deque of dicts -> dict of stacked arrays
        filtered_obs = {
            k: np.stack([d[k].detach().cpu().numpy() for d in ob_buffer], axis=0)
            for k in filtered_obs.keys()
        }
        """# 4. Prepare input for policy based on model type
        if context_length > 1:
            # For transformer models, use the entire buffer as sequence
            policy_input = ob_buffer.copy()  # Use all observations in buffer
            

            # Convert to proper format for transformer
            # Stack observations along sequence dimension
            stacked_obs = {}
            for key in training_obs_keys:
                if key in policy_input[0]:
                    # Stack all observations for this key: [context_length, obs_dim]
                    stacked_values = torch.stack([obs[key] for obs in policy_input])
                    # Add batch dimension: [1, context_length, obs_dim]
                    stacked_obs[key] = stacked_values.unsqueeze(0)
            
            filtered_obs = stacked_obs
            print("SHAPE OBS:--------------", filtered_obs)"""

        # Log full observations to terminal with detailed breakdown
        print(f"\n=== STEP {i} OBSERVATIONS ===")
        for key, value in filtered_obs.items():
            if torch.is_tensor(value):
                if key == "object":
                    print(f"{key}: (shape={value.shape})")
                    obj_data = value.cpu().numpy()
                    
                    # Parse the object observation structure (handle both 39 and 42+ element cases)
                    if len(obj_data) >= 39:  # Minimum expected: 3*7 + 6*3 = 39 elements
                        print("  📦 CUBE POSITIONS & ORIENTATIONS:")
                        print(f"    cube_1 pos:  [{obj_data[0]:.4f}, {obj_data[1]:.4f}, {obj_data[2]:.4f}]")
                        print(f"    cube_1 quat: [{obj_data[3]:.4f}, {obj_data[4]:.4f}, {obj_data[5]:.4f}, {obj_data[6]:.4f}]")
                        print(f"    cube_2 pos:  [{obj_data[7]:.4f}, {obj_data[8]:.4f}, {obj_data[9]:.4f}]")
                        print(f"    cube_2 quat: [{obj_data[10]:.4f}, {obj_data[11]:.4f}, {obj_data[12]:.4f}, {obj_data[13]:.4f}]")
                        print(f"    cube_3 pos:  [{obj_data[14]:.4f}, {obj_data[15]:.4f}, {obj_data[16]:.4f}]")
                        print(f"    cube_3 quat: [{obj_data[17]:.4f}, {obj_data[18]:.4f}, {obj_data[19]:.4f}, {obj_data[20]:.4f}]")
                        
                        print("  📏 RELATIVE DISTANCES:")
                        print(f"    gripper → cube_1: [{obj_data[21]:.4f}, {obj_data[22]:.4f}, {obj_data[23]:.4f}] (dist: {np.linalg.norm(obj_data[21:24]):.4f})")
                        print(f"    gripper → cube_2: [{obj_data[24]:.4f}, {obj_data[25]:.4f}, {obj_data[26]:.4f}] (dist: {np.linalg.norm(obj_data[24:27]):.4f})")
                        print(f"    gripper → cube_3: [{obj_data[27]:.4f}, {obj_data[28]:.4f}, {obj_data[29]:.4f}] (dist: {np.linalg.norm(obj_data[27:30]):.4f})")
                        
                        # Handle inter-cube distances (may not be present in 39-element version)
                        if len(obj_data) >= 39:
                            print(f"    cube_1 → cube_2:  [{obj_data[30]:.4f}, {obj_data[31]:.4f}, {obj_data[32]:.4f}] (dist: {np.linalg.norm(obj_data[30:33]):.4f})")
                            print(f"    cube_2 → cube_3:  [{obj_data[33]:.4f}, {obj_data[34]:.4f}, {obj_data[35]:.4f}] (dist: {np.linalg.norm(obj_data[33:36]):.4f})")
                            print(f"    cube_1 → cube_3:  [{obj_data[36]:.4f}, {obj_data[37]:.4f}, {obj_data[38]:.4f}] (dist: {np.linalg.norm(obj_data[36:39]):.4f})")
                        
                        # Show any extra data beyond expected 39 elements
                        if len(obj_data) > 39:
                            print(f"  ⚠️  EXTRA DATA: {obj_data[39:]}")
                            
                    else:
                        print(f"    ⚠️  WARNING: Object observation has {len(obj_data)} elements, expected 39+")
                        print(f"    Raw values: {obj_data}")
                        
                else:
                    # For other observations (eef_pos, eef_quat, gripper_pos)
                    if context_length == 1 and value.dim() == 3:
                        if key == "eef_pos":
                            pos = value.cpu().numpy()
                            print(f"{key}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] 🤖")
                        elif key == "eef_quat":
                            quat = value.cpu().numpy()
                            print(f"{key}: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}] 🔄")
                        elif key == "gripper_pos":
                            grip = value.cpu().numpy()
                            grip_state = "OPEN" if grip[0] > 0.03 else "CLOSED" if grip[0] < 0.01 else "PARTIAL"
                            print(f"{key}: [{grip[0]:.4f}] 🖐️ ({grip_state})")
                        else:
                            print(f"{key}: {value.cpu().numpy()}")
            else:
                print(f"{key}: {value}")
        print("=" * 60)

        # 2.1 Log observations with metadata if save_observations is True
        if save_observations:
            obs_entry = {
                "step": i,
                "timestamp": i * 0.05,  # 20Hz = 0.05s per step
            }
            
            # Convert tensors to numpy and add to log
            for key, value in filtered_obs.items():
                if torch.is_tensor(value):
                    obs_entry[key] = value.cpu().numpy()
                else:
                    obs_entry[key] = np.array(value)
            
            # Append the observation data to the obs_entry
            observation_log.append(obs_entry)

            # 2.2 Create CSV-formatted data entry
            csv_entry = {
                "trial": trial_id,
                "step": i,
                "timestamp": i * 0.05,
            }
            
            # Add eef_pos, eef_quat, gripper_pos
            for key, value in filtered_obs.items():
                if torch.is_tensor(value):
                    value_np = value.cpu().numpy()
                    
                    if key == "eef_pos":
                        csv_entry["eef_pos_x"] = value_np[0]
                        csv_entry["eef_pos_y"] = value_np[1]
                        csv_entry["eef_pos_z"] = value_np[2]
                    elif key == "eef_quat":
                        csv_entry["eef_quat_x"] = value_np[0]
                        csv_entry["eef_quat_y"] = value_np[1]
                        csv_entry["eef_quat_z"] = value_np[2]
                        csv_entry["eef_quat_w"] = value_np[3]
                    elif key == "gripper_pos":
                        csv_entry["gripper_pos"] = value_np[0]
                    elif key == "object" and len(value_np) >= 39:
                        # Cube positions
                        csv_entry["cube_1_pos_x"] = value_np[0]
                        csv_entry["cube_1_pos_y"] = value_np[1]
                        csv_entry["cube_1_pos_z"] = value_np[2]
                        csv_entry["cube_1_quat_x"] = value_np[3]
                        csv_entry["cube_1_quat_y"] = value_np[4]
                        csv_entry["cube_1_quat_z"] = value_np[5]
                        csv_entry["cube_1_quat_w"] = value_np[6]
                        
                        csv_entry["cube_2_pos_x"] = value_np[7]
                        csv_entry["cube_2_pos_y"] = value_np[8]
                        csv_entry["cube_2_pos_z"] = value_np[9]
                        csv_entry["cube_2_quat_x"] = value_np[10]
                        csv_entry["cube_2_quat_y"] = value_np[11]
                        csv_entry["cube_2_quat_z"] = value_np[12]
                        csv_entry["cube_2_quat_w"] = value_np[13]
                        
                        csv_entry["cube_3_pos_x"] = value_np[14]
                        csv_entry["cube_3_pos_y"] = value_np[15]
                        csv_entry["cube_3_pos_z"] = value_np[16]
                        csv_entry["cube_3_quat_x"] = value_np[17]
                        csv_entry["cube_3_quat_y"] = value_np[18]
                        csv_entry["cube_3_quat_z"] = value_np[19]
                        csv_entry["cube_3_quat_w"] = value_np[20]
                        
                        # Relative distances
                        csv_entry["gripper_to_cube_1_x"] = value_np[21]
                        csv_entry["gripper_to_cube_1_y"] = value_np[22]
                        csv_entry["gripper_to_cube_1_z"] = value_np[23]
                        csv_entry["gripper_to_cube_1_dist"] = np.linalg.norm(value_np[21:24])
                        
                        csv_entry["gripper_to_cube_2_x"] = value_np[24]
                        csv_entry["gripper_to_cube_2_y"] = value_np[25]
                        csv_entry["gripper_to_cube_2_z"] = value_np[26]
                        csv_entry["gripper_to_cube_2_dist"] = np.linalg.norm(value_np[24:27])
                        
                        csv_entry["gripper_to_cube_3_x"] = value_np[27]
                        csv_entry["gripper_to_cube_3_y"] = value_np[28]
                        csv_entry["gripper_to_cube_3_z"] = value_np[29]
                        csv_entry["gripper_to_cube_3_dist"] = np.linalg.norm(value_np[27:30])
                        
                        # Inter-cube distances
                        csv_entry["cube_1_to_cube_2_x"] = value_np[30]
                        csv_entry["cube_1_to_cube_2_y"] = value_np[31]
                        csv_entry["cube_1_to_cube_2_z"] = value_np[32]
                        csv_entry["cube_1_to_cube_2_dist"] = np.linalg.norm(value_np[30:33])
                        
                        csv_entry["cube_2_to_cube_3_x"] = value_np[33]
                        csv_entry["cube_2_to_cube_3_y"] = value_np[34]
                        csv_entry["cube_2_to_cube_3_z"] = value_np[35]
                        csv_entry["cube_2_to_cube_3_dist"] = np.linalg.norm(value_np[33:36])
                        
                        csv_entry["cube_1_to_cube_3_x"] = value_np[36]
                        csv_entry["cube_1_to_cube_3_y"] = value_np[37]
                        csv_entry["cube_1_to_cube_3_z"] = value_np[38]
                        csv_entry["cube_1_to_cube_3_dist"] = np.linalg.norm(value_np[36:39])
            
            csv_data.append(csv_entry)
        # 3. Neural Network Inference / Policy Forward Pass
        actions = policy(filtered_obs)


        print("Actions Shape: ",actions.shape)
        # 4. Action Post-Processing (only if normalization was used)
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Log actions too (if enabled)
        if save_observations:
            observation_log[-1]["action"] = actions.cpu().numpy().flatten()
            # Also add actions to CSV data
            action_np = actions.cpu().numpy().flatten()
            for j, action_val in enumerate(action_np):
                csv_data[-1][f"action_{j}"] = action_val

        # Apply actions to the environment
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        # Get the next observation
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj, observation_log, csv_data
        elif terminated or truncated:
            return False, traj, observation_log, csv_data

    return False, traj, observation_log, csv_data


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    
    # Load the Isaac Lab environment for the specified task
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set terminations
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device -> Running neural network inference on approapriate device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load robomimic policy (BC,LSTMGMM, etc.) from checkpoint
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)
    # Read context_length from the policy configuration
    
    # Run policy
    results = []
    all_successful_observations = []
    all_csv_data = []
    
    # Run multiple independent rollouts for statistical evaluation
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        # Single episode rollout
        terminated, traj, observation_log, csv_data = rollout(
            policy, env, success_term, args_cli.horizon, device, 
            save_observations=args_cli.save_observations, trial_id=trial
        )
        results.append(terminated)
        
        # Collect successful observations if --save_observations is enabled
        # Only save data from successful trials
        if terminated and args_cli.save_observations and observation_log is not None:
            all_successful_observations.append({
                "trial": trial,
                "observations": observation_log,
                "success": True,
                "metadata": {
                    "task": args_cli.task,
                    "checkpoint": args_cli.checkpoint,
                    "horizon": args_cli.horizon,
                    "frequency_hz": 20,
                    "dt": 0.05,
                    "total_steps": len(observation_log),
                    "norm_factor_min": args_cli.norm_factor_min,
                    "norm_factor_max": args_cli.norm_factor_max
                }
            })
            
            # Collect CSV data from successful trials
            if csv_data is not None:
                all_csv_data.extend(csv_data)
        
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    # Save all successful observations (if enabled and collected)
    if args_cli.save_observations and all_successful_observations:
        # Get the directory from the CSV output file path
        import os
        csv_dir = os.path.dirname(args_cli.csv_output_file)
        csv_basename = os.path.splitext(os.path.basename(args_cli.csv_output_file))[0]
        
        # Construct PKL and JSON file paths in the same directory as CSV
        pkl_file = os.path.join(csv_dir, f"{csv_basename}.pkl")
        json_file = os.path.join(csv_dir, f"{csv_basename}.json")
        
        # Save as pickle
        with open(pkl_file, "wb") as f:
            pickle.dump(all_successful_observations, f)
        print(f"\nSaved {len(all_successful_observations)} successful trajectories to {pkl_file}")
        
        # Save CSV data
        if all_csv_data:
            df = pd.DataFrame(all_csv_data)
            df.to_csv(args_cli.csv_output_file, index=False)
            print(f"Saved detailed observations to CSV: {args_cli.csv_output_file}")
            print(f"CSV contains {len(df)} rows with columns: {list(df.columns)}")
        
        # Also save as JSON for easier inspection
        json_data = []
        for traj_data in all_successful_observations:
            json_traj = {
                "trial": traj_data["trial"], 
                "metadata": traj_data["metadata"],
                "observations": []
            }
            for obs in traj_data["observations"]:
                json_obs = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        json_obs[key] = value.tolist()
                    else:
                        json_obs[key] = value
                json_traj["observations"].append(json_obs)
            json_data.append(json_traj)
        
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Also saved as JSON: {json_file}")
        
        # Print summary statistics
        lengths = [len(traj["observations"]) for traj in all_successful_observations]
        print(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
        
        # Print sample observation shapes for verification
        if all_successful_observations:
            sample_obs = all_successful_observations[0]["observations"][0]
            print(f"\nSample observation shapes:")
            for key, value in sample_obs.items():
                if key not in ["step", "timestamp"]:
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
    elif args_cli.save_observations and not all_successful_observations:
        print(f"\n[WARNING] No successful trajectories to save. Success rate: {results.count(True) / len(results)}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()