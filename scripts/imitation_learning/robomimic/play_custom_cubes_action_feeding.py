# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic with custom cube configurations.

This script loads a robomimic policy and plays it in an Isaac Lab environment,
cycling through multiple custom cube configurations.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, override the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
    custom_cube_poses: JSON file path with custom cube configurations.
    save_observations: If provided, save successful observations to file.
    obs_output_file: Output file path for saved observations.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import json 
import numpy as np
import pandas as pd
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy with custom cube configurations.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

# Custom cube configuration arguments
parser.add_argument("--custom_cube_poses", type=str, required=True, 
                   help="JSON file path with custom cube configurations")

# Arguments for observation logging
parser.add_argument("--save_observations", action="store_true", default=False, 
                   help="Save successful observations to file.")
parser.add_argument("--obs_output_file", type=str, default="custom_cube_observations.pkl", 
                   help="Output file path for saved observations.")
parser.add_argument("--csv_output_file", type=str, default="custom_cube_observations.csv", 
                   help="Output CSV file path for detailed observations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
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


def load_custom_cube_configurations(file_path):
    """Load custom cube configurations from JSON file.
    
    Args:
        file_path: Path to JSON file containing cube configurations
        
    Returns:
        List of cube configurations, each containing poses for 3 cubes
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Support both new multi-configuration format and legacy single configuration
        if isinstance(data, dict) and "configurations" in data:
            # New format: {"configurations": [{"name": ..., "poses": [...]}, ...]}
            configurations = data["configurations"]
            print(f"[INFO] Loaded {len(configurations)} cube configurations:")
            for i, config in enumerate(configurations):
                print(f"  {i}: {config['name']} - {config.get('description', 'No description')}")
            return configurations
        elif isinstance(data, list):
            # Legacy format: [{"pos": ..., "quat": ...}, ...]
            print(f"[INFO] Converting legacy format to new configuration format")
            legacy_config = {
                "name": "legacy_config",
                "description": "Converted from legacy format",
                "poses": data
            }
            return [legacy_config]
        else:
            print(f"[ERROR] Invalid JSON format in {file_path}")
            return None
            
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {file_path}: {e}")
        return None


def setup_cube_configuration(env_cfg, cube_poses):
    """Setup environment configuration with custom cube poses.
    
    Args:
        env_cfg: Environment configuration
        cube_poses: List of cube poses [{"pos": [x,y,z], "quat": [x,y,z,w]}, ...]
        
    Returns:
        Modified environment configuration
    """
    # Create a copy to avoid modifying the original
    import copy
    env_cfg_copy = copy.deepcopy(env_cfg)
    
    # Disable random cube positioning
    if hasattr(env_cfg_copy, 'events') and hasattr(env_cfg_copy.events, 'randomize_cube_positions'):
        env_cfg_copy.events.randomize_cube_positions = None
        print("[INFO] Disabled random cube position randomization")
    
    # Add custom cube positioning event
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
    
    if not hasattr(env_cfg_copy, 'events'):
        from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import EventCfg
        env_cfg_copy.events = EventCfg()
    
    env_cfg_copy.events.set_custom_cube_poses = EventTerm(
        func=franka_stack_events.set_fixed_object_poses,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
            "fixed_poses": cube_poses,
        },
    )
    
    return env_cfg_copy


def update_environment_events(env, cube_poses):
    """Update the environment's event configuration with new cube poses.
    
    Args:
        env: The Isaac Lab environment
        cube_poses: List of cube poses to set
    """
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
    
    # Update the environment's event configuration
    if not hasattr(env.cfg, 'events'):
        from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import EventCfg
        env.cfg.events = EventCfg()
    
    # Remove any existing custom pose event
    if hasattr(env.cfg.events, 'set_custom_cube_poses'):
        delattr(env.cfg.events, 'set_custom_cube_poses')
    
    # Add new custom cube positioning event
    env.cfg.events.set_custom_cube_poses = EventTerm(
        func=franka_stack_events.set_fixed_object_poses,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
            "fixed_poses": cube_poses,
        },
    )
    
    # Re-initialize the event manager with the new configuration
    from isaaclab.managers import EventManager
    env.event_manager = EventManager(env.cfg.events, env)
    
    print(f"[INFO] Updated environment events with new cube poses")


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
    sequence_length = 8  # T
    B = 1  
    policy.start_episode()
    obs_dict, _ = env.reset()
   
    # Initialize trajectory storage
    traj = dict(actions=[], obs=[], next_obs=[])

    # Add observation logging 
    observation_log = [] if save_observations else None
    csv_data = [] if save_observations else None

    training_obs_keys = ["eef_pos", "eef_quat", "gripper_pos", "object"]
                   # batch size

    # Buffer for one observation group (e.g., "policy")
    observation_buffer = {key: [] for key in training_obs_keys}
    for i in range(horizon):
        # 1. Observation Preprocessing
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # 2. Filter observations
        filtered_obs = {key: obs[key] for key in training_obs_keys if key in obs}
        # 3. Append new obs (real timestep) 
        if sequence_length > 1:
            for key in filtered_obs:
                # remove extra batch dim if present
                """new_obs = filtered_obs[key]
                if new_obs.ndim > 1:
                    new_obs = new_obs.squeeze(0)  # shape [D]
                observation_buffer[key].append(new_obs)
                
                if len(observation_buffer[key]) > sequence_length:
                    observation_buffer[key].pop(0)"""
                new_obs = filtered_obs[key]
                if new_obs.ndim > 1:
                    new_obs = new_obs.squeeze(0)  # shape [D]
                observation_buffer[key].insert(0, new_obs)

                if len(observation_buffer[key]) > sequence_length:
                    observation_buffer[key].pop(-1)

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
            
        """obs_seq = {}
        for k in obs_buffer[0].keys():
            arrs = []
            for o in obs_buffer:
                val = o[k]
                val = np.concatenate([
                    v.cpu().numpy().flatten() if torch.is_tensor(v) else np.asarray(v, dtype=np.float32).flatten()
                    for v in val.values()
                ])
                # If val is a dict, flatten or select the correct subfield
                if isinstance(val, dict):
                    # Example: flatten all values (if all are numeric)
                    val = np.concatenate([np.asarray(v, dtype=np.float32).flatten() for v in val.values()])
                else:
                    val = np.asarray(val, dtype=np.float32)
                arrs.append(val)
            stacked = np.stack(arrs, axis=0)
            if stacked.ndim == 1:
                stacked = stacked[:, None] 
            obs_seq[k] = stacked[None, ...]
            obs_seq[k] = torch.from_numpy(obs_seq[k]).float().to(device)"""

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
    """Run a trained policy from robomimic with custom cube configurations."""
    
    # Load custom cube configurations
    cube_configurations = load_custom_cube_configurations(args_cli.custom_cube_poses)
    if not cube_configurations:
        print("[ERROR] Failed to load cube configurations. Exiting.")
        return
    
    # Load the Isaac Lab environment for the specified task
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None


    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Initialize configuration cycling
    current_config_index = 0
    
    # Run policy
    results = []
    all_successful_observations = []
    all_csv_data = []
    
    # Acquire device and load policy
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)
    
    # Create environment once with the first configuration
    first_config = cube_configurations[0]
    initial_env_cfg = setup_cube_configuration(env_cfg, first_config["poses"])
    env = gym.make(args_cli.task, cfg=initial_env_cfg).unwrapped
    print(f"[INFO] Environment created with initial configuration: {first_config['name']}")
    
    # Run multiple rollouts, cycling through configurations
    for trial in range(args_cli.num_rollouts):
        # Select current configuration
        current_config = cube_configurations[current_config_index]
        current_poses = current_config["poses"]
        
        print(f"\n[INFO] Trial {trial}/{args_cli.num_rollouts}")
        print(f"[INFO] Using configuration: {current_config['name']}")
        print(f"[INFO] Description: {current_config.get('description', 'N/A')}")
        
        # Print cube positions for verification
        print(f"[INFO] Expected cube positions:")
        for i, pose in enumerate(current_poses):
            pos = pose["pos"]
            print(f"  Cube {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # For trials after the first, update the environment's event configuration
        if trial > 0:
            print(f"[INFO] Updating environment events for configuration: {current_config['name']}")
            update_environment_events(env, current_poses)
        
        # Set seed (different for each trial to add variation)
        torch.manual_seed(args_cli.seed + trial)
        env.seed(args_cli.seed + trial)
        
        # Run rollout
        terminated, traj, observation_log, csv_data = rollout(
            policy, env, success_term, args_cli.horizon, device, 
            save_observations=args_cli.save_observations, trial_id=trial, 
            config_info=current_config
        )
        results.append(terminated)
        
        # Collect successful observations
        if terminated and args_cli.save_observations and observation_log is not None:
            trial_data = {
                "trial": trial,
                "observations": observation_log,
                "success": True,
                "cube_configuration": current_config,
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
            }
            all_successful_observations.append(trial_data)
            
            # Add CSV data
            if csv_data is not None:
                all_csv_data.extend(csv_data)
        
        # Cycle to next configuration
        current_config_index = (current_config_index + 1) % len(cube_configurations)
        
        next_config_name = cube_configurations[current_config_index]['name']
        print(f"[INFO] Next configuration: {next_config_name}")

    # Print results summary
    success_count = results.count(True)
    success_rate = success_count / len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Successful trials: {success_count}/{len(results)}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Trial results: {results}")
    
    # Configuration-wise success analysis
    config_successes = {}
    for i, (trial_result, config_idx) in enumerate(zip(results, range(len(results)))):
        config_name = cube_configurations[config_idx % len(cube_configurations)]['name']
        if config_name not in config_successes:
            config_successes[config_name] = {'success': 0, 'total': 0}
        config_successes[config_name]['total'] += 1
        if trial_result:
            config_successes[config_name]['success'] += 1
    
    print(f"\nConfiguration-wise success rates:")
    for config_name, stats in config_successes.items():
        rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {config_name}: {stats['success']}/{stats['total']} ({rate:.2%})")

    # Save observations
    if args_cli.save_observations and all_successful_observations:
        import os
        
        # Create output directory if it doesn't exist
        csv_dir = os.path.dirname(args_cli.csv_output_file) or "."
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_basename = os.path.splitext(os.path.basename(args_cli.csv_output_file))[0]
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
            print(f"CSV contains {len(df)} rows with {len(df.columns)} columns")
        
        # Save as JSON
        json_data = []
        for traj_data in all_successful_observations:
            json_traj = {
                "trial": traj_data["trial"], 
                "cube_configuration": traj_data["cube_configuration"],
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
        if lengths:
            print(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
    
    elif args_cli.save_observations and not all_successful_observations:
        print(f"\n[WARNING] No successful trajectories to save. Success rate: {success_rate:.2%}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()