# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""


"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--bottleneck",
    default=False,
    help="Botttleneck Augmentation",
)
parser.add_argument(
    "--backward_augment",
    action="store_true",
    help="Enable backward-augment pre-processing of the loaded demo pool.",
)
parser.add_argument(
    "--ba_num_aug_per_episode",
    type=int,
    default=2,
    help="How many augmented snippets to synthesize per eligible episode.",
)
parser.add_argument(
    "--ba_r_min",
    type=float,
    default=0.20,
    help="Min spherical offset radius (meters) from the bottleneck pose.",
)
parser.add_argument(
    "--ba_r_max",
    type=float,
    default=0.50,
    help="Max spherical offset radius (meters) from the bottleneck pose.",
)
parser.add_argument(
    "--ba_rot_deg",
    type=float,
    default=7.5,
    help="Rotation step size (deg) when constructing reverse rotations.",
)
parser.add_argument(
    "--ba_balance_by_state",
    action="store_true",
    help="Inverse-frequency sampling over critical states (helps balance).",
)
    
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import gymnasium as gym
import inspect
import numpy as np
import random
import torch

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv


import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_mimic.datagen.generation_juicer import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401


def main():
    # Parse command line argument
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    env_name = args_cli.task or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        backward_augment=args_cli.backward_augment,
    )
    # Create environment -> The environment uses your current domain randomization
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # reset before starting
    env.reset()



    #-------------------------------------------------
    # Prepare asynchronous data generation using IsaacLab MIMIC framework
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        bottleneck=args_cli.bottleneck,
        backward_augment=args_cli.backward_augment,
        ba_num_aug_per_episode=args_cli.ba_num_aug_per_episode,
        ba_r_min=args_cli.ba_r_min,
        ba_r_max=args_cli.ba_r_max,
        ba_rot_deg=args_cli.ba_rot_deg,
        ba_balance_by_state=args_cli.ba_balance_by_state,
    )
    print(async_components)

    # Start main data generation loop, which interacts with the env and the mimic data generation logic
    if async_components is not None:
        print("Starting async data generation...")
        try:
            print("----------------------------------------------------")
            asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
            env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
    else:
        env.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
