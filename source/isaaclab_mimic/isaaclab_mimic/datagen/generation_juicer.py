# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import torch
from typing import Any
import numpy as np
from typing import List, Dict
from scipy.spatial.transform import Rotation as R, Slerp
from isaaclab_mimic.datagen.datagen_info import DatagenInfo

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_mimic.datagen.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory
from isaaclab_tasks.manager_based.manipulation.stack.mdp.terminations import eef_close_to_bottleneck_franka

from isaaclab_mimic.datagen.data_generator_juicer_2 import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# global variable to keep track of the data generation
num_success = 0
num_failures = 0
num_attempts = 0


async def run_data_generator(
    env: ManagerBasedRLMimicEnv,
    env_id: int,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term: TerminationTermCfg,
    pause_subtask: bool = False,
    backward_augment: bool = False,
    bottleneck: bool = False

):
    """Run mimic data generation from the given data generator in the specified environment index.

    Args:
        env: The environment to run the data generator on.
        env_id: The environment index to run the data generation on.
        env_reset_queue: The asyncio queue to send environment (for this particular env_id) reset requests to.
        env_action_queue: The asyncio queue to send actions to for executing actions.
        data_generator: The data generator instance to use.
        success_term: The success termination term to use.
        pause_subtask: Whether to pause the subtask during generation.
    """
    global num_success, num_failures, num_attempts
    while True:
        results=None
        if not backward_augment:
            results = await data_generator.generate(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue,
                pause_subtask=pause_subtask
            )
        else:
            results = await data_generator.generate_backward_augment(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue
            )

        if bool(results["success"]):
            num_success += 1
        else:
            num_failures += 1
        num_attempts += 1


def env_loop(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    shared_datagen_info_pool: DataGenInfoPool,
    asyncio_event_loop: asyncio.AbstractEventLoop,
):
    """Main asyncio loop for the environment.

    Args:
        env: The environment to run the main step loop on.
        env_reset_queue: The asyncio queue to handle reset request the environment.
        env_action_queue: The asyncio queue to handle actions to for executing actions.
        shared_datagen_info_pool: The shared datagen info pool that stores source demo info.
        asyncio_event_loop: The main asyncio event loop.
    """
    global num_success, num_failures, num_attempts
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:

            # check if any environment needs to be reset while waiting for actions
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            actions = torch.zeros(env.action_space.shape)

            # get actions from all the data generators
            for i in range(env.num_envs):
                # an async-blocking call to get an action from a data generator
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            # perform action on environment
            env.step(actions)

            # mark done so the data generators can continue with the step results
            for i in range(env.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != num_attempts:
                prev_num_attempts = num_attempts
                generated_sucess_rate = 100 * num_success / num_attempts if num_attempts > 0 else 0.0
                print("")
                print("*" * 50, "\033[K")
                print(
                    f"{num_success}/{num_attempts} ({generated_sucess_rate:.1f}%) successful demos generated by"
                    " mimic\033[K"
                )
                print("*" * 50, "\033[K")

                # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
                generation_guarantee = env.cfg.datagen_config.generation_guarantee
                generation_num_trials = env.cfg.datagen_config.generation_num_trials
                check_val = num_success if generation_guarantee else num_attempts
                if check_val >= generation_num_trials:
                    print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                    break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

    env.close()


def setup_env_config(
    env_name: str,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_num_trials: int | None = None,
    backward_augment: bool = False,
) -> tuple[Any, Any]:
    env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
    print(f"Loaded environment config for {env_cfg.terminations}")

    if generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = generation_num_trials

    env_cfg.env_name = env_name

    # Extract success checking function
    if backward_augment:
        # Use your custom termination function for backward augment
        success_term = TerminationTermCfg(
            func=eef_close_to_bottleneck_franka,
            params={
                #"eef_name": "panda_hand",  # or your EEF name
                "pos_tol": 0.01,
                #"ori_tol_deg": 5.0,
                #"use_gripper": False,
            }
        )
        print("chooosen")

        
    elif hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # Configure for data generation
    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    # Setup recorders
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def setup_async_generation(
    env: Any,
    num_envs: int,
    input_file: str,
    success_term: Any,
    pause_subtask: bool = False,
    bottleneck: bool = False,
    backward_augment: bool = False,
    ba_num_aug_per_episode: int = 2,
    ba_r_min: float = 0.20,
    ba_r_max: float = 0.50,
    ba_rot_deg: float = 7.5,
    ba_balance_by_state: bool = False,
) -> dict[str, Any]:
    """Setup async data generation tasks.

    Args:
        env: The environment instance
        num_envs: Number of environments to run
        input_file: Path to input dataset file
        success_term: Success termination condition
        pause_subtask: Whether to pause after subtasks

    Returns:
        List of asyncio tasks for data generation
    """
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()
    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(env, env.cfg, env.device, asyncio_lock=shared_datagen_info_pool_lock)
    shared_datagen_info_pool.load_from_dataset_file(input_file)
    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")
    #print(shared_datagen_info_pool.datagen_infos[0].subtask_term_signals["bottlenecks"])
    # In setup_async_generation function
    """if backward_augment:
        print("[BA] Running enhanced backward-augment with critical state detection...")
        num_added = apply_backward_augmentation_to_pool(
            env=env,
            info_pool=shared_datagen_info_pool,
            num_aug_per_episode=ba_num_aug_per_episode,
            r_min=ba_r_min,
            r_max=ba_r_max,
            rot_step_deg=ba_rot_deg,
            balance_by_state=ba_balance_by_state,
            device=env.device,
        )
        print(f"[BA] Added {num_added} augmented trajectories with critical state validation.")
        
        # Save the augmented data pool if needed
        #         
        # Return None to signal that no mimic generation should occur
        return None
        
        # Normal mimic generation setup (only runs if backward_augment=False)
        # Create and schedule data generator tasks"""
    data_generator = DataGenerator(env=env, src_demo_datagen_info_pool=shared_datagen_info_pool)
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        task = asyncio_event_loop.create_task(
            run_data_generator(
                env, i, env_reset_queue, env_action_queue, data_generator, success_term, pause_subtask=pause_subtask, backward_augment=backward_augment
            )
        )
        data_generator_asyncio_tasks.append(task)

    return {
         "tasks": data_generator_asyncio_tasks,
        "event_loop": asyncio_event_loop,
        "reset_queue": env_reset_queue,
        "action_queue": env_action_queue,
        "info_pool": shared_datagen_info_pool,
    }
