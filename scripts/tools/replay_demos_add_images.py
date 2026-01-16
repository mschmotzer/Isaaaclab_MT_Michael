# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""
"""Launch Isaac Sim Simulator first."""

import argparse
import shutil
import h5py

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib
import gymnasium as gym
import os
import torch

from isaaclab.devices import Se3Keyboard
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape mismatch: {state_name} of {asset_name}")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'State ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] mismatch\n'
    return states_matched, output_log


def main():
    global is_paused

    # -------------------------------
    # CREATE OUTPUT HDF5 FILE COPY
    # -------------------------------
    src_file = args_cli.dataset_file
    if not os.path.exists(src_file):
        raise FileNotFoundError(src_file)

    dst_file = src_file.replace(".hdf5", "_new_obs.hdf5")
    shutil.copy(src_file, dst_file)
    h5_out = h5py.File(dst_file, "r+")  # We will overwrite /data/<episode>/obs/*
    print(f"\n📁 Created output file: {dst_file}\n")

    # load dataset file handler
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(src_file)

    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        print("No episodes found.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes or list(range(episode_count))
    if args_cli.task:
        env_name = args_cli.task

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)

    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env_cfg.events = {}

    env = gym.make(env_name, cfg=env_cfg).unwrapped

    teleop = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
    teleop.add_callback("N", play_cb)
    teleop.add_callback("B", pause_cb)

    validate_state = args_cli.validate_states and args_cli.num_envs == 1

    idle_action = getattr(env_cfg, "idle_action", torch.zeros(env.action_space.shape))
    if hasattr(idle_action, "repeat"):
        idle_action = idle_action.repeat(args_cli.num_envs, 1)

    env.reset()
    teleop.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {i: EpisodeData() for i in range(args_cli.num_envs)}
            first_loop = True
            has_next_action = True

            while has_next_action:
                actions = idle_action.clone()
                has_next_action = False

                for env_id in range(args_cli.num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()

                    if env_next_action is None:
                        if not episode_indices_to_replay:
                            continue

                        next_ep_idx = episode_indices_to_replay.pop(0)
                        if next_ep_idx >= episode_count:
                            continue

                        replayed_episode_count += 1
                        ep_name = episode_names[next_ep_idx]
                        print(f"{replayed_episode_count:4}: Loading episode #{next_ep_idx} into env_{env_id}")

                        episode_data = dataset_file_handler.load_episode(ep_name, env.device)
                        env_episode_data_map[env_id] = episode_data

                        initial_state = episode_data.get_initial_state()
                        if initial_state:
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=False)
                        else:
                            env.reset(torch.tensor([env_id], device=env.device))

                        env_next_action = episode_data.get_next_action()
                        has_next_action = True

                    if env_next_action is not None:
                        actions[env_id] = env_next_action
                        has_next_action = True

                if not first_loop:
                    while is_paused:
                        env.sim.render()
                        continue
                first_loop = False

                obs, _, _, _, info = env.step(actions)

                # ---------------------------------------------------------
                # WRITE UPDATED OBS TO THE NEW FILE (PER-TIMESTEP)
                # ---------------------------------------------------------
                for env_id in range(args_cli.num_envs):

                    episode_data = env_episode_data_map[env_id]
                    if episode_data.next_action_index == 0:
                        continue  # episode just loaded, no step yet

                    step_idx = episode_data.next_action_index - 1
                    ep_name = episode_names[next_ep_idx]

                    obs_group = h5_out["data"][ep_name]["obs"]

                    # Write each key from obs["policy"]
                    print(obs.keys())
                    for key, value in obs["rgb_camera"].items():
                        data_np = value.cpu().numpy()
                        # Create dataset only once
                        print(key)
                        if key not in obs_group:
                            ep_len = data_np.shape[0]
                            obs_dim = data_np.shape[-1]
                            obs_group.create_dataset(
                                key,
                                shape=(ep_len, obs_dim),
                                dtype=data_np.dtype,
                            )
                            print("KEYY:", key)
                            obs_group[key][step_idx] = data_np[env_id]

                # ----------------------------
                # STATE VALIDATION IF ENABLED
                # ----------------------------
                if validate_state:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        runtime_state = env.scene.get_state(is_relative=True)
                        matched, log = compare_states(state_from_dataset, runtime_state, 0)
                        if matched:
                            print(f"State at {env_episode_data_map[0].next_state_index - 1} matched.")
                        else:
                            print(log)

            break

    print(f"Finished replaying {replayed_episode_count} episodes.")
    h5_out.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
