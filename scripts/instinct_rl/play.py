"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import subprocess
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop headless playback after this many steps.")
parser.add_argument(
    "--grail_diagnostic_csv",
    type=str,
    default=None,
    help="Write per-step GRAIL stair posture and foot-contact diagnostics to a new CSV file.",
)
parser.add_argument(
    "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
)
parser.add_argument(
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be positive")
if args_cli.grail_diagnostic_csv is not None and args_cli.max_steps is None:
    parser.error("--grail_diagnostic_csv requires --max_steps")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from instinct_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.managers.reward_manager import MultiRewardManager
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


GRAIL_DIAGNOSTIC_COLUMNS = [
    "env_id",
    "episode_step",
    "reference_time_s",
    "projected_gravity_error",
    "base_x",
    "base_y",
    "base_z",
    "reference_base_x",
    "reference_base_y",
    "reference_base_z",
    "left_ankle_x",
    "left_ankle_y",
    "left_ankle_z",
    "right_ankle_x",
    "right_ankle_y",
    "right_ankle_z",
    "left_contact_force_n",
    "right_contact_force_n",
    "env_origin_x",
    "env_origin_y",
    "env_origin_z",
    "done_after_action",
    "base_pg_too_far_after_action",
    "link_pos_too_far_after_action",
    "dataset_exhausted_after_action",
]


def _grail_diagnostic_snapshot(env, robot_foot_ids: list[int], contact_foot_ids: list[int]) -> list[list[float]]:
    """Capture state before the action so automatic episode resets cannot erase the failure approach."""
    scene = env.unwrapped.scene
    robot = scene["robot"]
    reference = scene["motion_reference"]
    contact_sensor = scene["contact_forces"]
    env_ids = reference.ALL_INDICES

    reference_time_s = (
        reference.complete_motion_lengths - reference.assigned_motion_lengths
        + env.unwrapped.episode_length_buf * env.unwrapped.step_dt
    )
    reference_base = reference.data.base_pos_w[env_ids, reference.aiming_frame_idx]
    reference_quat = reference.data.base_quat_w[env_ids, reference.aiming_frame_idx]
    robot_pg = math_utils.quat_apply_inverse(robot.data.root_state_w[:, 3:7], robot.data.GRAVITY_VEC_W)
    reference_pg = math_utils.quat_apply_inverse(reference_quat, robot.data.GRAVITY_VEC_W)
    pg_error = torch.linalg.vector_norm(robot_pg - reference_pg, dim=-1)
    ankle_positions = robot.data.body_pos_w[:, robot_foot_ids].reshape(env.unwrapped.num_envs, 6)
    contact_forces = (
        contact_sensor.data.net_forces_w_history[:, :, contact_foot_ids]
        .norm(dim=-1)
        .max(dim=1)
        .values
    )

    values = torch.cat(
        (
            env_ids.float().unsqueeze(-1),
            env.unwrapped.episode_length_buf.float().unsqueeze(-1),
            reference_time_s.unsqueeze(-1),
            pg_error.unsqueeze(-1),
            robot.data.root_pos_w,
            reference_base,
            ankle_positions,
            contact_forces,
            reference.env_origins,
        ),
        dim=-1,
    )
    return values.cpu().tolist()


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # react to custom play arguments
    if args_cli.no_terminate:
        # NOTE: This is only applicable with shadowing task
        env.unwrapped.termination_manager._term_cfgs = [
            env.unwrapped.termination_manager._term_cfgs[
                env.unwrapped.termination_manager._term_names.index("dataset_exhausted")
            ]
        ]
        env.unwrapped.termination_manager._term_names = ["dataset_exhausted"]

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.export_as_onnx(obs, export_model_dir)

    diagnostic_file = None
    if args_cli.grail_diagnostic_csv is not None:
        if args_cli.no_terminate:
            raise ValueError("GRAIL diagnostics require the normal termination conditions.")
        robot_foot_ids, _ = env.unwrapped.scene["robot"].find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        contact_foot_ids, _ = env.unwrapped.scene["contact_forces"].find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        if len(robot_foot_ids) != 2 or len(contact_foot_ids) != 2:
            raise RuntimeError("Both ankle links must be present in the robot and contact sensor.")
        diagnostic_file = open(args_cli.grail_diagnostic_csv, "x", newline="")
        diagnostic_writer = csv.writer(diagnostic_file)
        diagnostic_writer.writerow(GRAIL_DIAGNOSTIC_COLUMNS)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                diagnostic_rows = (
                    _grail_diagnostic_snapshot(env, robot_foot_ids, contact_foot_ids)
                    if diagnostic_file is not None
                    else None
                )
                # agent stepping
                actions = policy(obs)
                if timestep < args_cli.zero_act_until:
                    actions[:] = 0.0
                # env stepping
                obs, rewards, dones, infos = env.step(actions)
                if diagnostic_rows is not None:
                    done_flags = dones.bool().cpu().tolist()
                    reason_flags = {
                        name: env.unwrapped.termination_manager.get_term(name).bool().cpu().tolist()
                        for name in ("base_pg_too_far", "link_pos_too_far", "dataset_exhausted")
                    }
                    for env_id, row in enumerate(diagnostic_rows):
                        done = done_flags[env_id]
                        diagnostic_writer.writerow(
                            row
                            + [
                                int(done),
                                int(done and reason_flags["base_pg_too_far"][env_id]),
                                int(done and reason_flags["link_pos_too_far"][env_id]),
                                int(done and reason_flags["dataset_exhausted"][env_id]),
                            ]
                        )
            timestep += 1

            # override reward terms if auxiliary reward is enabled
            if args_cli.aux_reward:
                # NOTE: This is only applicable when reward term has `.reward` to be overridden
                aux_rewards = ppo_runner.alg.compute_auxiliary_reward(infos["observations"])
                for aux_reward_name, aux_reward in aux_rewards.items():
                    aux_term_cfg = env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)  # type: ignore
                    aux_term_cfg.func.reward[:] = aux_reward * getattr(ppo_runner.alg, aux_reward_name + "_coef", 1.0)

            if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                break
            if args_cli.video and timestep >= args_cli.video_length:
                break
    finally:
        if diagnostic_file is not None:
            diagnostic_file.close()
            print(f"[INFO] GRAIL diagnostic CSV: {args_cli.grail_diagnostic_csv}")
        env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
