# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Replay recorded demonstrations and save updated observations including RGB images.

This script:
- Loads an existing dataset of demonstrations.
- Replays episodes in the Isaac Lab environment.
- Records RGB camera observations in addition to existing state data.
- Saves a new dataset file with augmented observations.
"""

import argparse
import contextlib
import os
import h5py
import numpy as np
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher
from isaaclab.devices import Se3Keyboard
from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# --------------------------------------------------------
# Command-line arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="Replay demonstrations with RGB and save to a new dataset.")
parser.add_argument("--dataset_file", type=str, required=True, help="Path to source dataset (HDF5).")
parser.add_argument("--task", type=str, required=True, help="Task name to replay (RGB-enabled).")
parser.add_argument("--output_file", type=str, default=None, help="Output dataset file with RGB.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--validate_states", action="store_true", help="Validate states against source dataset.")
parser.add_argument("--enable_pinocchio", action="store_true", help="Enable Pinocchio if required.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --------------------------------------------------------
# Launch simulator
# --------------------------------------------------------
if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------
# Load dataset
# --------------------------------------------------------
if not os.path.exists(args_cli.dataset_file):
    raise FileNotFoundError(f"Dataset file {args_cli.dataset_file} does not exist.")

dataset_handler = HDF5DatasetFileHandler()
dataset_handler.open(args_cli.dataset_file)
episode_count = dataset_handler.get_num_episodes()
episode_names = list(dataset_handler.get_episode_names())
if episode_count == 0:
    raise ValueError("No episodes found in dataset.")

# Output file
output_path = args_cli.output_file or args_cli.dataset_file.replace(".hdf5", "_with_rgb.hdf5")
output_file = h5py.File(output_path, "w")
output_file.attrs["env_name"] = args_cli.task
output_file.attrs["source_dataset"] = args_cli.dataset_file

# --------------------------------------------------------
# Load environment configuration (RGB)
# --------------------------------------------------------
if "RGB" not in args_cli.task:
    args_cli.task += "RGB"  # Force RGB environment

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.recorders = {}
env_cfg.terminations = {}
env_cfg.events = {}

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

# Verify camera exists
if not hasattr(env.scene, "camera0"):
    raise ValueError("Camera 'camera0' not found in the scene. Make sure the environment is RGB-enabled.")

# Teleop interface (needed for stepping)
teleop = Se3Keyboard()
teleop.reset()

# --------------------------------------------------------
# Helper: validate states
# --------------------------------------------------------
def compare_states(state_from_dataset, runtime_state, runtime_env_idx):
    states_matched = True
    log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_val = runtime_state[asset_type][asset_name][state_name][runtime_env_idx]
                dataset_val = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_val) != len(runtime_val):
                    raise ValueError(f"State shape mismatch: {state_name} for {asset_name}")
                for i in range(len(dataset_val)):
                    if abs(dataset_val[i] - runtime_val[i]) > 0.01:
                        states_matched = False
                        log += f"{asset_type}/{asset_name}/{state_name}[{i}] mismatch: dataset={dataset_val[i]}, runtime={runtime_val[i]}\n"
    return states_matched, log

# --------------------------------------------------------
# Replay episodes
# --------------------------------------------------------
for ep_idx, ep_name in enumerate(episode_names):
    print(f"Replaying episode {ep_idx+1}/{episode_count}: {ep_name}")
    episode_data = dataset_handler.load_episode(ep_name, env.device)

    # Reset environment to initial state
    initial_state = episode_data.get_initial_state()
    if initial_state is not None:
        env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=False)
    else:
        env.reset(torch.tensor([0], device=env.device))

    actions = episode_data.actions
    rgb_images = []

    # Step through episode
    for t in range(len(actions)):
        action = actions[t:t+1]  # single-env batch
        env.step(action)

        # Capture RGB
        rgb_img = env.scene.camera0.data.output["rgb"][0].cpu().numpy()  # HWC
        rgb_images.append(rgb_img)

        # Optional: validate states
        if args_cli.validate_states and t < len(episode_data.states):
            dataset_state = episode_data.states[t]
            runtime_state = env.scene.get_state(is_relative=False)
            matched, log = compare_states(dataset_state, runtime_state, 0)
            if not matched:
                print(f"State mismatch at step {t}:\n{log}")

    # Save episode
    ep_group = output_file.create_group(ep_name)
    # RGB observations
    ep_group.create_dataset("observations/rgb", data=np.stack(rgb_images, axis=0), compression="gzip")
    # Copy original actions
    if episode_data.actions is not None:
        ep_group.create_dataset("actions", data=episode_data.actions.cpu().numpy())
    # Copy original states
    if episode_data.states is not None and len(episode_data.states) > 0:
        states_group = ep_group.create_group("states")
        for si, state in enumerate(episode_data.states):
            step_group = states_group.create_group(f"state_{si}")
            for atype, assets in state.items():
                atype_group = step_group.create_group(atype)
                for aname, asset_data in assets.items():
                    asset_group = atype_group.create_group(aname)
                    for key, val in asset_data.items():
                        asset_group.create_dataset(key, data=val.cpu().numpy())
    # Copy initial state
    if initial_state is not None:
        init_group = ep_group.create_group("initial_state")
        for atype, assets in initial_state.items():
            atype_group = init_group.create_group(atype)
            for aname, asset_data in assets.items():
                asset_group = atype_group.create_group(aname)
                for key, val in asset_data.items():
                    asset_group.create_dataset(key, data=val.cpu().numpy())

    print(f"Saved episode '{ep_name}' with {len(rgb_images)} RGB frames.")

# --------------------------------------------------------
# Cleanup
# --------------------------------------------------------
output_file.close()
dataset_handler.close()
env.close()
simulation_app.close()
print(f"All episodes replayed and saved to: {output_path}")
