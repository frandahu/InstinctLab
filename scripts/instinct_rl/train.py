# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import multiprocessing as mp
import os
import sys

from cudnn_bootstrap import preload_torch_cudnn

preload_torch_cudnn()

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--logroot", type=str, default=None, help="Override default log root path, typically `log/instinct_rl/`."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_root", help="Motion dataset root for Mixed Parkour or Curb Crossing.")
parser.add_argument("--motion_selection", help="Curated G1 motion YAML for Mixed Parkour or Curb Crossing.")
parser.add_argument("--warm_start", help="Curb crossing only: initialize policy/value weights, with fresh AMP and optimizers.")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Enable distributed training. No need to add manually, it will be set automatically in the script.",
)
parser.add_argument(
    "--local-rank",
    type=int,
    help="Local rank for distributed training. No need to add manually, it will be set automatically in the script.",
)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
# train.py specific arguments
parser.add_argument("--cprofile", action="store_true", default=False, help="Enable cProfile.")
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
motion_tasks = ("Instinct-Parkour-Mixed-Amp-G1-v0", "Instinct-Parkour-Curb-Crossing-Amp-G1-v0")
if (args_cli.motion_root or args_cli.motion_selection) and args_cli.task not in motion_tasks:
    parser.error("--motion_root/--motion_selection require a Mixed Parkour or Curb Crossing task")
if args_cli.warm_start and (args_cli.task != motion_tasks[1] or args_cli.resume):
    parser.error("--warm_start requires Curb Crossing and cannot be combined with --resume")
if "LOCAL_RANK" in os.environ:
    args_cli.distributed = True

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.distributed as dist
from datetime import datetime

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# set affinity of in multiprocessing
def auto_affinity():
    rank = int(os.environ["RANK"])  # Get rank assigned by torch
    num_cores = mp.cpu_count() // torch.cuda.device_count()
    core_range = range(rank * num_cores, (rank + 1) * num_cores)
    core_mask = ",".join(map(str, core_range))
    os.system(f"taskset -cp {core_mask} {os.getpid()}")
    print("Affinity auto updated to:", core_mask, "for rank:", rank)


@hydra_task_config(args_cli.task, "instinct_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: InstinctRlOnPolicyRunnerCfg):
    """Train with Instinct-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_instinct_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    motion_inventory = None
    if args_cli.task == "Instinct-Parkour-Curb-Crossing-Amp-G1-v0":
        from prepare_curb_crossing_motions import configure_curb_motions

        if args_cli.warm_start and agent_cfg.resume:
            raise ValueError("--warm_start cannot be combined with agent.resume=True")
        motion_inventory = configure_curb_motions(env_cfg, args_cli.motion_root, args_cli.motion_selection)
        print(
            f"[CURB] references={motion_inventory['files']}, selection={motion_inventory['selection']}, "
            f"AMP coef={agent_cfg.algorithm.discriminator_reward_coef}, warm_start={args_cli.warm_start}",
            flush=True,
        )
    if args_cli.task == "Instinct-Parkour-Mixed-Amp-G1-v0":
        from check_parkour_motions import configure_training_motions

        motion_inventory = configure_training_motions(env_cfg, args_cli.motion_root, args_cli.motion_selection)
        print(
            f"[MIXED] motions={motion_inventory['files']}, subsets={motion_inventory['subset_counts']}, "
            f"AMP reward coef={agent_cfg.algorithm.discriminator_reward_coef}, "
            f"resume={agent_cfg.resume}", flush=True,
        )
        print(f"[MIXED] motion root={motion_inventory['root']}, selection={motion_inventory['selection']}", flush=True)
        print("[MIXED] terrain proportions=" + str({
            name: cfg.proportion for name, cfg in env_cfg.scene.terrain.terrain_generator.sub_terrains.items()
        }), flush=True)

    if args_cli.task == "Instinct-Parkour-Stairs-Amp-G1-v1":
        # Keep task and auxiliary reward on the same physical-time scale, even
        # when simulation dt/decimation are changed through Hydra overrides.
        agent_cfg.algorithm.discriminator_reward_coef = env_cfg.amp_reward_rate * env_cfg.sim.dt * env_cfg.decimation
        env_cfg.scene.terrain.terrain_generator.seed = env_cfg.seed
        print(
            f"[STAIRS] step_dt={env_cfg.sim.dt * env_cfg.decimation:.6f}, "
            f"AMP reward coef={agent_cfg.algorithm.discriminator_reward_coef:.6f}, "
            f"resume={agent_cfg.resume}, policy={agent_cfg.policy.class_name}", flush=True
        )

    # prepare configs for distributed training
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            backend="nccl",
            rank=app_launcher.local_rank,
            world_size=int(os.getenv("WORLD_SIZE", 1)),
        )
        auto_affinity()
        local_rank, world_size = dist.get_rank(), dist.get_world_size()
        env_cfg.seed += local_rank
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        print(
            f"[INFO] Distributed training with rank: {local_rank}, world size: {world_size}, rank: {os.environ['RANK']}"
        )

    # specify directory for logging experiments
    if args_cli.task == "Instinct-Parkour-Curb-Crossing-Amp-G1-v0":
        env_cfg.scene.terrain.terrain_generator.seed = env_cfg.seed
    if args_cli.logroot is None:
        log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
    else:
        log_root_path = args_cli.logroot

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(env_cfg, "run_name", None):
        log_dir += f"_{env_cfg.run_name}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
        for h_args in hydra_args:
            log_dir += "_"
            log_dir += h_args.split("=")[0].split(".")[-1]
            log_dir += "-"
            log_dir += h_args.split("=")[1]
    log_dir = os.path.join(log_root_path, log_dir)

    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint)  # type: ignore
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming experiment from directory: {resume_path}")
        if motion_inventory is not None:
            from check_parkour_motions import check_resume_motion_inventory

            check_resume_motion_inventory(resume_path, motion_inventory)
        resume_run_name = os.path.basename(os.path.dirname(resume_path))
        log_dir += f"_from{resume_run_name.split('_')[0]}_{resume_run_name.split('_')[1]}"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # create runner from instinct-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.task in ("Instinct-Parkour-Stairs-Amp-G1-v1", *motion_tasks) and not (
        "LOCAL_RANK" in os.environ and dist.get_rank() > 0
    ):
        from stair_training_runtime import save_runtime_sources

        if motion_inventory is not None:
            save_runtime_sources(runner, env_cfg, log_dir, filename="parkour_runtime.json")
            with open(os.path.join(log_dir, "params", "motion_inventory.json"), "w", encoding="utf-8") as stream:
                json.dump(motion_inventory, stream, indent=2, ensure_ascii=False)
        else:
            save_runtime_sources(runner, env_cfg, log_dir)
    # # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif args_cli.warm_start:
        from curb_crossing_warm_start import warm_start

        initialization = warm_start(runner, args_cli.warm_start, agent_cfg)
        if not ("LOCAL_RANK" in os.environ and dist.get_rank() > 0):
            os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
            with open(os.path.join(log_dir, "params", "warm_start.json"), "w", encoding="utf-8") as stream:
                json.dump(initialization, stream, indent=2)
        print(f"[CURB] warm-start: {initialization}", flush=True)

    # dump the configuration into log-directory
    if not ("LOCAL_RANK" in os.environ and dist.get_rank() > 0):
        # prevent dumping the config in non-rank-0 process
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if args_cli.cprofile:
        import cProfile

        cprofile = cProfile.Profile()
        print(
            "Profiling enabled, a .profile file will be saved in the log directory after the program successfully"
            " finished."
        )
        cprofile.enable()

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
    )
    runner.save(os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"))

    if args_cli.cprofile:
        cprofile.disable()
        cprofile.dump_stats(os.path.join(log_dir, "cprofile_stats.profile"))

    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
