"""Evaluate a frozen depth-based G1 Parkour policy on reproducible straight stairs.

Run from the repository root. See eval_stairs.md for commands and metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from stair_eval_cases import classify_episode, make_cases, make_flat_control_cases, summarize
from terrain_eval_cases import make_terrain_cases
from terrain_eval_presets import apply_difficulty_defaults
from policy_diagnostics import DIAGNOSTIC_COLUMNS, policy_step_diagnostics
from mp4_video import Mp4Recorder, first_render_frame, video_camera_pose, video_overview_pose
from eval_startup import StartupDiagnostics
from training_yaml import load_training_yaml, restore_training_env_config


def build_parser(defaults=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Instinct-Parkour-Target-Amp-G1-v0")
    parser.add_argument("--load_run", required=True, help="Run directory, absolute or relative to logs/instinct_rl/g1_parkour")
    parser.add_argument("--checkpoint", required=True, help="Exact filename (e.g. model_2000.pt); no latest-model matching")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--episodes_per_env", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=("baseline", "hard"), default="baseline",
                        help="Evaluation-only preset; explicit CLI options override preset values")
    parser.add_argument("--stair_mode", choices=("up_down", "mixed", "up", "down"), default="up_down")
    parser.add_argument(
        "--terrain_mode", choices=("stairs", "flat", "slope", "curb", "mixed", "suite"), default="stairs",
        help="Stairs, flat control, slope, curb, continuous mixed route, or five-terrain suite",
    )
    parser.add_argument("--slope_angle_range", type=float, nargs=2, default=(5.0, 20.0), metavar=("MIN", "MAX"),
                        help="Ramp angle in degrees, for slope/mixed/suite")
    parser.add_argument("--ramp_length", type=float, default=2.0, help="Horizontal length of each ramp, m")
    parser.add_argument("--curb_height_range", type=float, nargs=2, default=(0.08, 0.20), metavar=("MIN", "MAX"))
    parser.add_argument("--curb_count", type=int, default=1, help="Number of successive curb platforms")
    parser.add_argument("--curb_depth_range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Platform length along travel; default uses landing_depth")
    parser.add_argument("--curb_gap_range", type=float, nargs=2, default=(1.0, 1.0), metavar=("MIN", "MAX"),
                        help="Ground distance after each curb platform, m")
    parser.add_argument("--num_steps", type=int, default=8, help="Steps per flight (up_down has this many up AND down)")
    parser.add_argument("--step_height_range", type=float, nargs=2, default=(0.08, 0.20), metavar=("MIN", "MAX"))
    parser.add_argument("--tread_depth_range", type=float, nargs=2, default=(0.25, 0.40), metavar=("MIN", "MAX"))
    parser.add_argument("--stair_width", type=float, default=2.0)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--irregular", action="store_true", dest="irregular", help="Sparse dimensional deviations (default)")
    geometry.add_argument("--regular", action="store_false", dest="irregular", help="All steps in a flight use nominal dimensions")
    parser.set_defaults(irregular=True)
    parser.add_argument("--vary_both_dimensions", action="store_true",
                        help="Change both height and tread depth at each selected irregular step")
    parser.add_argument("--irregular_fraction", type=float, default=0.20, help="Fraction of affected steps per flight; rounded up and capped below half")
    parser.add_argument("--dimension_variation", type=float, default=0.25, help="Maximum relative deviation from nominal dimensions, clipped to requested ranges")
    parser.add_argument("--landing_depth", type=float, default=1.20, help="Top platform depth between ascent and descent, m")
    parser.add_argument("--speed", type=float, default=0.5, help="Forward velocity limit, m/s")
    parser.add_argument("--command_mode", choices=("goal", "straight"), default="goal",
                        help="Goal navigation (default), or flat-only fixed world +X heading control")
    parser.add_argument("--centered_start", action="store_true",
                        help="Remove the small seeded XY/yaw spawn offsets for a controlled start")
    parser.add_argument(
        "--sample", action="store_true",
        help="Diagnostic only: sample policy actions instead of using their deterministic mean",
    )
    parser.add_argument("--episode_length_s", type=float, default=45.0)
    parser.add_argument("--success_hold_s", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, help="Optional global step limit; unfinished trials remain uncompleted")
    parser.add_argument("--trace_stride", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, help="Result directory; if it exists, create a new numbered sibling")
    parser.add_argument("--use_current_cfg", action="store_true", help="Explicitly use current task config instead of saved training configs")
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--video", action="store_true", dest="video", help="Write staircase.mp4 (default)")
    video.add_argument("--no_video", action="store_false", dest="video", help="Metrics only; disable RGB rendering/encoding")
    parser.set_defaults(video=True)
    parser.add_argument("--video_length", type=int, help="Optional recording limit in simulation steps; default records the full evaluation")
    parser.add_argument("--video_stride", type=int, default=2, help="Record every N simulation steps; default is 25 FPS at step_dt=0.02 s")
    parser.add_argument("--video_env_id", type=int, default=0, help="Lane shown in the fixed spectator camera")
    parser.add_argument("--video_view", choices=("lane", "overview"), default="lane",
                        help="Show one lane or all parallel robots in one MP4")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--dry_run", action="store_true", help="Write cases.json only; no Isaac Sim or checkpoint required")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(defaults=None):
    parser = build_parser(defaults)
    apply_difficulty_defaults(parser)
    numpy_runtime = None
    if "--dry_run" in sys.argv or "--help" in sys.argv or "-h" in sys.argv:
        # Permit a CPU-only preview on a machine without Isaac Lab.
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--enable_cameras", action="store_true")
        launcher_class = None
    else:
        from render_bootstrap import preload_isaac_numpy
        from cudnn_bootstrap import preload_torch_cudnn

        # OmniGraph/Replicator in Isaac Sim 5.x use the NumPy 1.x binary ABI.
        # Select it before either AppLauncher or PyTorch can import NumPy 2.x.
        numpy_runtime = preload_isaac_numpy()
        preload_torch_cudnn()
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        launcher_class = AppLauncher
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    args.numpy_runtime = numpy_runtime
    if args.command_mode == "straight" and args.terrain_mode != "flat":
        parser.error("--command_mode straight requires --terrain_mode flat")
    if args.terrain_mode in ("slope", "curb", "mixed", "suite") and args.stair_mode != "up_down":
        parser.error("slope/curb/mixed/suite use complete up-and-down obstacles; omit --stair_mode")
    if args.task not in (
        "Instinct-Parkour-Target-Amp-G1-v0", "Instinct-Parkour-Stairs-Amp-G1-v1",
        "Instinct-Parkour-Mixed-Amp-G1-v0",
    ):
        parser.error("This evaluator supports the G1 Parkour v0, Stairs v1, and Mixed v0 tasks")
    for name in ("episodes_per_env", "trace_stride", "video_stride"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.video_length is not None and args.video_length <= 0:
        parser.error("--video_length must be positive")
    if not 0 <= args.video_env_id < args.num_envs:
        parser.error("--video_env_id must be in [0, num_envs)")
    for name in ("video_width", "video_height"):
        if getattr(args, name) < 64 or getattr(args, name) % 2:
            parser.error(f"--{name} must be an even integer >= 64")
    for name in ("speed", "episode_length_s", "success_hold_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be finite and positive")
    if args.success_hold_s >= args.episode_length_s:
        parser.error("--success_hold_s must be less than --episode_length_s")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if Path(args.checkpoint).name != args.checkpoint or args.checkpoint in (".", ".."):
        parser.error("--checkpoint must be an exact filename inside --load_run")
    try:
        if args.terrain_mode in ("slope", "curb", "mixed", "suite"):
            cases = make_terrain_cases(
                args.num_envs, args.seed, args.terrain_mode, args.num_steps,
                tuple(args.step_height_range), tuple(args.tread_depth_range), args.stair_width, args.irregular,
                args.irregular_fraction, args.dimension_variation, args.landing_depth,
                tuple(args.slope_angle_range), args.ramp_length, tuple(args.curb_height_range),
                args.curb_count, args.curb_depth_range, tuple(args.curb_gap_range), args.vary_both_dimensions,
            )
        else:
            cases = make_cases(
                args.num_envs, args.seed, args.stair_mode, args.num_steps,
                tuple(args.step_height_range), tuple(args.tread_depth_range), args.stair_width, args.irregular,
                args.irregular_fraction, args.dimension_variation, args.landing_depth,
                args.vary_both_dimensions,
            )
    except ValueError as error:
        parser.error(str(error))
    if args.terrain_mode == "flat":
        cases = make_flat_control_cases(cases)
    args.device = args.device or "cuda:0"
    if args.video:
        args.enable_cameras = True
    return args, cases, launcher_class


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def create_output_directory(requested):
    """Reserve a fresh directory atomically, preserving all previous results."""
    requested = Path(requested).expanduser().resolve()
    candidate, suffix = requested, 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            # A parent-file conflict must be reported, not retried indefinitely.
            if not candidate.exists() and not candidate.is_symlink():
                raise
            suffix += 1
            candidate = requested.with_name(f"{requested.name}_{suffix:03d}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_weights(actor_critic, normalizers, agent_cfg, checkpoint):
    """Load only inference state via CPU, without AMP or optimizer state."""
    import torch

    if agent_cfg.get("ckpt_manipulator"):
        raise ValueError("Evaluation requires a directly loadable checkpoint; ckpt_manipulator is not supported")
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    actor_critic.load_state_dict(loaded["model_state_dict"], strict=True)
    for group, normalizer in normalizers.items():
        key = f"{group}_normalizer_state_dict"
        if key not in loaded:
            raise KeyError(f"Checkpoint is missing required normalizer: {key}")
        normalizer.load_state_dict(loaded[key])
    if agent_cfg.get("empirical_normalization", False) and not normalizers:
        raise ValueError("Legacy empirical normalization requires an explicit compatible inference loader")
    return loaded.get("iter", 0)


def build_inference_policy(env, agent_cfg, checkpoint, device, sample=False):
    """Construct the saved actor directly; WasabiPPO is training-only machinery."""
    from instinct_rl import modules
    from instinct_rl.utils.utils import get_subobs_size

    obs_format = env.get_obs_format()
    policy_cfg = agent_cfg["policy"].copy()
    actor_critic = modules.build_actor_critic(
        policy_cfg.pop("class_name"), policy_cfg, obs_format,
        num_actions=env.num_actions, num_rewards=env.num_rewards,
    ).to(device)
    normalizers = {}
    if "policy" in agent_cfg.get("normalizers", {}):
        cfg = agent_cfg["normalizers"]["policy"].copy()
        normalizers["policy"] = modules.build_normalizer(
            input_shape=get_subobs_size(obs_format["policy"]),
            normalizer_class_name=cfg.pop("class_name"), normalizer_kwargs=cfg,
        ).to(device)
    iteration = load_policy_weights(actor_critic, normalizers, agent_cfg, checkpoint)
    actor_critic.eval()
    for normalizer in normalizers.values():
        normalizer.eval()
    actor = actor_critic.act if sample else actor_critic.act_inference
    print(f"[INFO] Policy action mode: {'sampled' if sample else 'deterministic_mean'}", flush=True)
    if hasattr(actor_critic, "std"):
        import torch

        std = actor_critic.std.detach()
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Checkpoint action standard deviation must be finite and positive")
        print(
            f"[INFO] Checkpoint action std: mean={std.mean().item():.4f}, "
            f"min={std.min().item():.4f}, max={std.max().item():.4f}", flush=True,
        )
    if "policy" in normalizers:
        return lambda obs: actor(normalizers["policy"](obs)), iteration
    return actor, iteration


def validate_policy_observations(base):
    """Keep the checkpoint's actor inputs ordered and free of critic-only velocity."""
    expected = (
        "base_ang_vel", "projected_gravity", "velocity_commands",
        "joint_pos", "joint_vel", "actions", "depth_image",
    )
    actual = tuple(base.observation_manager.active_terms["policy"])
    if actual != expected:
        raise RuntimeError(f"Unsupported policy observation terms: {list(actual)}; expected {list(expected)}")


TRACE_COLUMNS = [
    "env_id", "episode_index", "episode_step", "elapsed_s", "base_x_m", "base_y_m", "base_z_m",
    "ground_height_m", "root_clearance_m", "projected_gravity_z", "vel_x_m_s", "vel_y_m_s", "yaw_rate_rad_s",
    "command_x_m_s", "command_y_m_s", "command_yaw_rad_s", "goal_distance_m",
    "left_ankle_x_m", "left_ankle_y_m", "left_ankle_z_m", "right_ankle_x_m", "right_ankle_y_m",
    "right_ankle_z_m", "left_contact_force_n", "right_contact_force_n", "summit_reached", "done", "termination_reasons",
] + ["heading_w_rad", "vel_x_world_m_s", "vel_y_world_m_s"] + DIAGNOSTIC_COLUMNS
TRACE_COLUMNS += ["terrain", "checkpoints_passed", "checkpoints_required"]
EPISODE_COLUMNS = [
    "env_id", "episode_index", "direction", "case_seed", "outcome", "elapsed_s", "steps",
    "max_progress_fraction", "max_abs_lateral_error_m", "mean_velocity_error_m_s", "termination_reasons",
    "spawn_x_offset_m", "spawn_y_offset_m", "spawn_yaw_offset_rad", "summit_reached",
    "terrain", "checkpoints_passed", "checkpoints_required",
]


def run_evaluation(args, cases, output, diagnostics):
    diagnostics.phase("import_evaluation_modules")
    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import parse_env_cfg

    import instinctlab.tasks  # noqa: F401 -- task registration
    import cli_args
    from instinctlab.tasks.parkour.config.g1.stair_eval_cfg import configure_stair_evaluation
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    diagnostics.phase("load_agent_config", args.task)
    agent_cfg = cli_args.parse_instinct_rl_cfg(args.task, args)
    diagnostics.phase("load_task_config", args.task)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    diagnostics.phase("resolve_checkpoint", args.load_run)
    run_dir = Path(args.load_run).expanduser()
    if not run_dir.is_absolute():
        run_dir = Path("logs/instinct_rl") / agent_cfg.experiment_name / run_dir
    checkpoint = (run_dir / args.checkpoint).resolve(strict=True)
    env_path, agent_path = run_dir / "params/env.yaml", run_dir / "params/agent.yaml"
    config_restore = None
    if args.use_current_cfg:
        agent_dict = agent_cfg.to_dict()
    else:
        if not env_path.is_file() or not agent_path.is_file():
            raise FileNotFoundError(
                f"Saved training params missing under {run_dir / 'params'}. "
                "Use --use_current_cfg only if the current observation/action/network config matches training."
            )
        diagnostics.phase("load_saved_env", str(env_path))
        saved_env = load_training_yaml(env_path)
        diagnostics.phase("apply_saved_env")
        # The evaluation generator replaces training terrain geometry entirely.
        # Restore saved sensors/actions/physics, omitting this superseded subtree.
        config_restore = restore_training_env_config(
            env_cfg, saved_env, skip_paths=("/scene/terrain/terrain_generator",)
        )
        print(
            f"[INFO] Restored saved environment: initialized {len(config_restore['initialized_optional_fields'])} "
            "optional fields; training terrain generator will be replaced by evaluation routes.", flush=True
        )
        diagnostics.phase("load_saved_agent", str(agent_path))
        agent_dict = load_training_yaml(agent_path)
    diagnostics.phase("configure_stairs")
    env_cfg = configure_stair_evaluation(
        env_cfg, cases, args.seed, args.speed, args.episode_length_s, args.success_hold_s,
        command_mode=args.command_mode, centered_start=args.centered_start,
    )
    env_cfg.sim.device = args.device
    if args.video:
        selected = cases[args.video_env_id]
        lane_y = (args.video_env_id - (args.num_envs - 1) / 2) * env_cfg.scene.terrain.terrain_generator.size[1]
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (
            video_overview_pose(cases, env_cfg.scene.terrain.terrain_generator.size[1])
            if args.video_view == "overview" else video_camera_pose(selected, lane_y)
        )
        env_cfg.viewer.resolution = (args.video_width, args.video_height)
    agent_dict["device"] = args.device
    diagnostics.phase("hash_checkpoint", str(checkpoint))
    manifest = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    manifest.update({
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "config_source": "current_checkout" if args.use_current_cfg else "saved_training_params",
        "numpy_runtime": args.numpy_runtime,
        "saved_config_restore": config_restore,
        "saved_env_sha256": sha256(env_path) if env_path.is_file() else None,
        "saved_agent_sha256": sha256(agent_path) if agent_path.is_file() else None,
        "step_dt_s": env_cfg.sim.dt * env_cfg.decimation,
    })
    write_json(output / "cases.json", manifest)
    diagnostics.phase("save_eval_configs")
    dump_yaml(str(output / "eval_env.yaml"), env_cfg)
    dump_yaml(str(output / "eval_agent.yaml"), agent_dict)
    diagnostics.phase("create_environment")
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    episodes = []
    counters = [0] * args.num_envs
    progress = [0.0] * args.num_envs
    lateral = [0.0] * args.num_envs
    velocity_error = [0.0] * args.num_envs
    status, error_text = "initializing", None
    total_steps = 0
    recorder = None
    video_error = None
    try:
        diagnostics.phase("reset_vector_environment")
        env = InstinctRlVecEnvWrapper(env)
        base = env.unwrapped
        validate_policy_observations(base)
        diagnostics.phase("load_policy_weights", str(checkpoint))
        policy, iteration = build_inference_policy(env, agent_dict, checkpoint, args.device, sample=args.sample)
        observation, _ = env.get_observations()
        policy_format = env.get_obs_format()["policy"]
        previous_actions = None
        print(f"[INFO] Loaded inference actor from iteration {iteration}; no motion dataset or AMP runner.", flush=True)
        success_term = base.termination_manager.get_term_cfg("stair_success").func
        if args.video:
            diagnostics.phase("initialize_mp4_encoder")
            video_name = "terrain_walk.mp4" if any("profile" in case for case in cases) else "staircase.mp4"
            recorder = Mp4Recorder(output / video_name, fps=1.0 / (base.step_dt * args.video_stride))
            diagnostics.phase("render_first_frame")
            recorder.append(first_render_frame(base))
            print(f"[INFO] Off-screen MP4: {recorder.path}, lane={args.video_env_id}, fps={recorder.fps:g}", flush=True)
        max_steps = args.max_steps or base.max_episode_length * args.episodes_per_env
        print(f"[INFO] Checkpoint: {checkpoint}")
        print(f"[INFO] {args.num_envs} lanes, {args.episodes_per_env} trials/lane; results: {output}")
        status = "running"
        diagnostics.phase("evaluate_policy")
        with (output / "trace.csv").open("x", newline="", encoding="utf-8") as trace_file, (
            output / "episodes.csv"
        ).open("x", newline="", encoding="utf-8") as episode_file:
            trace_writer = csv.writer(trace_file)
            trace_writer.writerow(TRACE_COLUMNS)
            episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_COLUMNS)
            episode_writer.writeheader()
            with torch.inference_mode():
                while total_steps < max_steps and simulation_app.is_running():
                    if not torch.isfinite(observation).all():
                        raise RuntimeError("Nonfinite policy observation; stopping before advancing physics")
                    actions = policy(observation)
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Nonfinite policy action; stopping before advancing physics")
                    action_diagnostics = policy_step_diagnostics(observation, actions, previous_actions, policy_format)
                    previous_actions = actions.detach().clone()
                    observation, _, dones, _ = env.step(actions)
                    previous_actions[dones.bool()] = 0.0
                    total_steps += 1
                    if (
                        recorder is not None and total_steps % args.video_stride == 0
                        and (args.video_length is None or total_steps <= args.video_length)
                    ):
                        recorder.append(base.render())
                    snapshot = {key: value.cpu().tolist() for key, value in success_term.snapshot.items()}
                    step_diagnostics = {key: value.cpu().tolist() for key, value in action_diagnostics.items()}
                    for key in ("joint_velocity_rms_rad_s", "applied_torque_rms_nm"):
                        step_diagnostics[key] = snapshot[key]
                    flags = {
                        name: base.termination_manager.get_term(name).bool().cpu().tolist()
                        for name in base.termination_manager.active_terms
                    }
                    done_flags = dones.bool().cpu().tolist()
                    for env_id, case in enumerate(cases):
                        if counters[env_id] >= args.episodes_per_env:
                            continue
                        step = int(snapshot["step"][env_id])
                        pos = snapshot["pos"][env_id]
                        velocity = snapshot["velocity"][env_id]
                        command = snapshot["command"][env_id]
                        finite = all(math.isfinite(value) for value in pos + velocity + command)
                        if finite:
                            progress[env_id] = max(progress[env_id], min(1.0, max(0.0, pos[0] / case["goal_x_m"])))
                            lateral[env_id] = max(lateral[env_id], abs(pos[1]))
                            velocity_error[env_id] += math.hypot(velocity[0] - command[0], velocity[1] - command[1])
                        reasons = [name for name, values in flags.items() if values[env_id]]
                        done = done_flags[env_id]
                        elapsed_s = step * base.step_dt
                        if step % args.trace_stride == 0 or done:
                            trace_writer.writerow([
                                env_id, counters[env_id], step, elapsed_s, *pos,
                                snapshot["floor"][env_id], snapshot["clearance"][env_id], snapshot["gravity_z"][env_id],
                                *velocity[:2], snapshot["yaw_rate"][env_id], *command,
                                snapshot["goal_distance"][env_id],
                                *snapshot["feet"][env_id][0], *snapshot["feet"][env_id][1],
                                *snapshot["foot_forces"][env_id], int(snapshot["summit_reached"][env_id]), int(done), "|".join(reasons),
                                snapshot["heading_w"][env_id], *snapshot["velocity_w"][env_id][:2],
                                *[step_diagnostics[key][env_id] for key in DIAGNOSTIC_COLUMNS],
                                case.get("terrain", "flat" if case["direction"] == "flat" else "stairs"),
                                snapshot["checkpoints_passed"][env_id], snapshot["checkpoints_required"][env_id],
                            ])
                        if done:
                            episode = {
                                "env_id": env_id, "episode_index": counters[env_id], "direction": case["direction"],
                                "case_seed": case["case_seed"],
                                "outcome": classify_episode({name: values[env_id] for name, values in flags.items()}),
                                "elapsed_s": elapsed_s, "steps": step,
                                "max_progress_fraction": progress[env_id], "max_abs_lateral_error_m": lateral[env_id],
                                "mean_velocity_error_m_s": velocity_error[env_id] / step if finite and step else None,
                                "termination_reasons": "|".join(reasons),
                                "spawn_x_offset_m": snapshot["spawn_offsets"][env_id][0],
                                "spawn_y_offset_m": snapshot["spawn_offsets"][env_id][1],
                                "spawn_yaw_offset_rad": snapshot["spawn_offsets"][env_id][2],
                                "summit_reached": snapshot["summit_reached"][env_id],
                                "terrain": case.get("terrain", "flat" if case["direction"] == "flat" else "stairs"),
                                "checkpoints_passed": snapshot["checkpoints_passed"][env_id],
                                "checkpoints_required": snapshot["checkpoints_required"][env_id],
                            }
                            episodes.append(episode)
                            episode_writer.writerow(episode)
                            episode_file.flush()
                            counters[env_id] += 1
                            progress[env_id] = lateral[env_id] = velocity_error[env_id] = 0.0
                    if total_steps % 250 == 0 or all(count >= args.episodes_per_env for count in counters):
                        print(f"[INFO] step={total_steps}, completed trials={len(episodes)}/{args.num_envs * args.episodes_per_env}", flush=True)
                        lane = args.video_env_id
                        detail = {key: values[lane] for key, values in step_diagnostics.items()}
                        print(
                            f"[DIAG] env={lane}, x={snapshot['pos'][lane][0]:.3f} m, "
                            f"y={snapshot['pos'][lane][1]:.3f} m, "
                            f"heading={math.degrees(snapshot['heading_w'][lane]):.1f} deg, "
                            f"vx_body={snapshot['velocity'][lane][0]:.3f}, vy_body={snapshot['velocity'][lane][1]:.3f} m/s, "
                            f"command_x={snapshot['command'][lane][0]:.3f} m/s, "
                            f"command_yaw={snapshot['command'][lane][2]:.3f} rad/s, "
                            f"observed_command_x=[{detail['observed_command_x_min_m_s']:.3f}, "
                            f"{detail['observed_command_x_max_m_s']:.3f}], "
                            f"action_rms={detail['policy_action_rms']:.4f}, "
                            f"action_delta_rms={detail['policy_action_delta_rms']:.4f}, "
                            f"joint_velocity_rms={detail['joint_velocity_rms_rad_s']:.4f} rad/s, "
                            f"torque_rms={detail['applied_torque_rms_nm']:.3f} Nm, "
                            f"depth_input=[{detail['depth_input_min']:.3f}, {detail['depth_input_max']:.3f}]",
                            flush=True,
                        )
                    if all(count >= args.episodes_per_env for count in counters):
                        status = "complete"
                        break
                if status != "complete":
                    status = "step_limit" if total_steps >= max_steps else "simulation_closed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status, error_text = "error", str(error)
        diagnostics.failure(error)
        raise
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception as error:
                video_error = str(error)
                status = "error"
                error_text = error_text or video_error
        report = summarize(episodes, args.num_envs * args.episodes_per_env)
        report.update({"status": status, "error": error_text, "simulation_steps": total_steps})
        report["action_mode"] = "sampled" if args.sample else "deterministic_mean"
        report["command_mode"] = args.command_mode
        report["centered_start"] = args.centered_start
        report["video"] = {
            "enabled": args.video,
            "path": str(recorder.path) if recorder is not None else None,
            "frames": recorder.frames if recorder is not None else 0,
            "fps": recorder.fps if recorder is not None else None,
            "env_id": args.video_env_id,
            "view": args.video_view,
            "encoding_error": video_error,
        }
        report["by_direction"] = {
            direction: summarize(
                [row for row in episodes if row["direction"] == direction],
                sum(case["direction"] == direction for case in cases) * args.episodes_per_env,
            )
            for direction in ("up_down", "up", "down", "flat") if any(case["direction"] == direction for case in cases)
        }
        report["by_terrain"] = {
            terrain: summarize(
                [row for row in episodes if row["terrain"] == terrain],
                sum(case.get("terrain", "flat" if case["direction"] == "flat" else "stairs") == terrain
                    for case in cases) * args.episodes_per_env,
            )
            for terrain in sorted({case.get("terrain", "flat" if case["direction"] == "flat" else "stairs")
                                   for case in cases})
        }
        write_json(output / "summary.json", report)
        print(f"[RESULT] status={status}, outcomes={report['outcome_counts']}, summary={output / 'summary.json'}")
        env.close()
        if video_error:
            raise RuntimeError(f"MP4 finalization failed: {video_error}")


def main(defaults=None):
    global simulation_app
    args, cases, launcher_class = parse_args() if defaults is None else parse_args(defaults)
    # The parser helper in this repository expects the standard runner flags.
    args.resume, args.run_name = True, None
    is_course = any("profile" in case for case in cases)
    requested_output = (
        args.output_dir or Path("outputs/terrain_eval" if is_course else "outputs/stair_eval")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    ).expanduser().resolve()
    output = create_output_directory(requested_output)
    if output != requested_output:
        print(f"[INFO] Output directory already exists; using a new directory: {output}", flush=True)
    print(f"[INFO] Results directory: {output}", flush=True)
    write_json(output / "cases.json", {
        "protocol": (
            "terrain_routes_v2" if is_course and (args.curb_count > 1 or args.vary_both_dimensions)
            else "terrain_routes_v1" if is_course else "straight_stairs_v3"
        ), "seed": args.seed, "cases": cases,
        "output_dir": str(output),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "unknown_terrain_definition": (
            "Seeded straight routes of stairs, ramps and curbs; training-range disjointness is not claimed."
            if is_course else
            "Flat locomotion control with the same route length and goals."
            if getattr(args, "terrain_mode", "stairs") == "flat" else
            "Novel stair routes with sparse dimensional deviations; training-range disjointness is not claimed."
        ),
        "repeats": "Each lane repeats the same geometry with seeded small spawn perturbations.",
    })
    if args.dry_run:
        print(f"[INFO] Generated {len(cases)} cases without simulation: {output / 'cases.json'}")
        return
    with StartupDiagnostics(output) as diagnostics:
        diagnostics.phase("launch_isaac_sim")
        app_launcher = launcher_class(args)
        simulation_app = app_launcher.app
        try:
            run_evaluation(args, cases, output, diagnostics)
            diagnostics.finish()
        except BaseException as error:
            diagnostics.failure(error)
            raise
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
