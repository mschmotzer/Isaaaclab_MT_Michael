


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


def rollout(policy, env, success_term, horizon, device, save_observations=False, trial_id=0, config_info=None):
    """Perform a single rollout of the policy in the environment.
   
    Args:
        policy: The robomimic policy to play.
        env: The environment to play in.
        success_term: The success termination condition.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.
        save_observations: Whether to save observations for logging.
        trial_id: The trial number for this rollout.
        config_info: Dictionary containing configuration information.

    Returns:
        terminated: Whether the rollout terminated successfully.
        traj: The trajectory of the rollout.
        observation_log: List of observations if save_observations is True, else None.
        csv_data: List of CSV-formatted observation data if save_observations is True, else None.
    """
    # Episode initialization 
    sequence_length = 2 # T
    B = 1  
    policy.start_episode()
    obs_dict, _ = env.reset()
   
    # Initialize trajectory storage
    traj = dict(actions=[], obs=[], next_obs=[])

    # Add observation logging 
    observation_log = [] if save_observations else None
    csv_data = [] if save_observations else None

    training_obs_keys = ["eef_pos", "eef_quat", "gripper_pos", "actions", "image","image2"]
                   # batch size

    # Buffer for one observation group (e.g., "policy")
    observation_buffer = {key: [] for key in training_obs_keys}
    for i in range(horizon):
        # 1. Observation Preprocessing
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # 2. Filt   er observations
        filtered_obs = {key: obs[key] for key in training_obs_keys if key in obs}
        
        # Add image data if needed
        if "image" in training_obs_keys and "rgb_camera" in obs_dict and "image" in obs_dict["rgb_camera"]:
            filtered_obs["image"] = obs_dict["rgb_camera"]["image"]
        # 3. Append new obs (real timestep) 
        if sequence_length > 0:
            for key in filtered_obs:
                # remove extra batch dim if present
                new_obs = filtered_obs[key]
                if new_obs.ndim > 1:
                    new_obs = new_obs.squeeze(0)  # shape [D]
                observation_buffer[key].append(new_obs)
                
                if len(observation_buffer[key]) > sequence_length:
                    observation_buffer[key].pop(0)

                #print(new_obs)
            # Only stack if buffer is full
            inputs = None
            if all(len(observation_buffer[k]) == sequence_length for k in observation_buffer):
                # Stack each modality along time
                inputs = {
                    key: torch.stack(observation_buffer[key], dim=0)#.unsqueeze(0)  # [1, T, D_key]
                    for key in training_obs_keys
                    if key in observation_buffer
                }
            #print("Inputs created", {k: v.shape for k, v in inputs.items()})
            #print("inputs: ", inputs)   
        else:
            inputs = filtered_obs
            


        # Log observations with configuration info
        if i == 0 and config_info:  # Only log on first step to avoid spam
            print(f"\n=== TRIAL {trial_id} - CONFIGURATION: {config_info['name']} ===")
            print(f"Description: {config_info.get('description', 'N/A')}")
            print("Initial cube positions:")
            for j, pose in enumerate(config_info['poses']):
                print(f"  Cube {j+1}: pos=[{pose['pos'][0]:.3f}, {pose['pos'][1]:.3f}, {pose['pos'][2]:.3f}]")

        # Log observations (reduced verbosity)
        if i % 50 == 0:  # Log every 50 steps instead of every step
            print(f"Step {i}: EEF pos=[{filtered_obs['eef_pos'].cpu().numpy()}], "
                  f"Gripper=[{filtered_obs['gripper_pos'].cpu().numpy()[0]:.3f}]")

        # 2.1 Log observations with metadata (if enabled)
        if save_observations:
            obs_entry = {
                "step": i,
                "timestamp": i * 0.05,  # 20Hz = 0.05s per step
                "configuration": config_info['name'] if config_info else "unknown",
            }
            
            # Convert tensors to numpy and add to log
            for key, value in filtered_obs.items():
                if torch.is_tensor(value):
                    obs_entry[key] = value.cpu().numpy()
                else:
                    obs_entry[key] = np.array(value)
            
            observation_log.append(obs_entry)

            # 2.2 Create CSV-formatted data entry
            csv_entry = {
                "trial": trial_id,
                "step": i,
                "timestamp": i * 0.05,
                "config_name": config_info['name'] if config_info else "unknown",
                "config_description": config_info.get('description', '') if config_info else "",
            }
            
            # Add detailed observation data to CSV
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
                        # Add cube positions and orientations
                        for cube_idx in range(3):
                            base_idx = cube_idx * 7
                            csv_entry[f"cube_{cube_idx+1}_pos_x"] = value_np[base_idx]
                            csv_entry[f"cube_{cube_idx+1}_pos_y"] = value_np[base_idx+1]
                            csv_entry[f"cube_{cube_idx+1}_pos_z"] = value_np[base_idx+2]
                            csv_entry[f"cube_{cube_idx+1}_quat_x"] = value_np[base_idx+3]
                            csv_entry[f"cube_{cube_idx+1}_quat_y"] = value_np[base_idx+4]
                            csv_entry[f"cube_{cube_idx+1}_quat_z"] = value_np[base_idx+5]
                            csv_entry[f"cube_{cube_idx+1}_quat_w"] = value_np[base_idx+6]
                        
                        # Add relative distances
                        distance_mappings = [
                            ("gripper_to_cube_1", 21), ("gripper_to_cube_2", 24), ("gripper_to_cube_3", 27),
                            ("cube_1_to_cube_2", 30), ("cube_2_to_cube_3", 33), ("cube_1_to_cube_3", 36)
                        ]
                        
                        for dist_name, start_idx in distance_mappings:
                            if start_idx + 2 < len(value_np):
                                csv_entry[f"{dist_name}_x"] = value_np[start_idx]
                                csv_entry[f"{dist_name}_y"] = value_np[start_idx+1]
                                csv_entry[f"{dist_name}_z"] = value_np[start_idx+2]
                                csv_entry[f"{dist_name}_dist"] = np.linalg.norm(value_np[start_idx:start_idx+3])
            
            csv_data.append(csv_entry)
        if inputs is not None:
            # 3. Neural Network Inference
            #print(f"Step {i}: Computing action")
            actions = policy(inputs)
            # 4. Action Post-Processing (only if normalization was used)
            if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
                actions = (
                    (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
                ) / 2 + args_cli.norm_factor_min

            actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

            # Log actions
            if save_observations:
                observation_log[-1]["action"] = actions.cpu().numpy().flatten()
                # Add actions to CSV data
                action_np = actions.cpu().numpy().flatten()
                for j, action_val in enumerate(action_np):
                    csv_data[-1][f"action_{j}"] = action_val

            # Apply actions to the environment
            #print(f"Step {i}: Applying actions: {actions.cpu().numpy().flatten()}")
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"]

            # Record trajectory
            traj["actions"].append(actions.tolist())
            traj["next_obs"].append(obs)

            # Check if rollout was successful
            if bool(success_term.func(env, **success_term.params)[0]):
                print(f"✅ Trial {trial_id} successful at step {i}")
                return True, traj, observation_log, csv_data
            elif terminated or truncated:
                print(f"❌ Trial {trial_id} failed at step {i}")
                return False, traj, observation_log, csv_data

    print(f"⏱️ Trial {trial_id} reached horizon without success")
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
    # Read sequence_length from the policy configuration
    
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