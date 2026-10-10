"""Frozen-policy foot-state demo. All new behavior stays in this directory.

Run from the repository root; see README.md. Simulator imports and NumPy occur
only after the existing Isaac Sim 5.1 bootstrap selects the runtime ABI.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "instinct_rl"))
evaluation = importlib.import_module("eval_stairs")


def parse_args():
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--sole_offset", type=float, nargs=3, default=(0.039, 0, -0.058), metavar=("X", "Y", "Z"))
    extra.add_argument("--velocity_cutoff_hz", type=float, default=10.0)
    extra.add_argument("--force_on_n", type=float, default=20.0)
    extra.add_argument("--force_off_n", type=float, default=10.0)
    extra.add_argument("--confirm_on_s", type=float, default=0.02)
    extra.add_argument("--confirm_off_s", type=float, default=0.02)
    extra.add_argument("--encoder_noise_std", type=float, default=0.0, help="Observer-only joint-angle noise, rad")
    extra.add_argument("--curb_layout", choices=("alternating", "sampled"), default="alternating",
                       help="Demo-only alternating near/far gaps, or evaluator's sampled gap range")
    extra.add_argument("--curb_gaps", type=float, nargs="+", default=(0.18, 0.85, 0.22, 1.0),
                       help="Inter-platform gaps in m, cycled; final ground exit is 1 m")
    reference_defaults = {
        "sole_half_length_m": 0.093, "sole_half_width_m": 0.036, "edge_margin_m": 0.025,
        "nominal_step_m": 0.32, "step_width_m": 0.20, "max_step_m": 0.55,
        "max_height_change_m": 0.24, "bridge_gap_m": 0.30, "surface_tolerance_m": 0.04,
        "anchor_motion_m": 0.015, "anchor_rotation_rad": 0.20,
    }
    for name, default in reference_defaults.items():
        extra.add_argument("--" + name, type=float, default=default)
    foot, remaining = extra.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        print("Independent foot-state options:\n" + extra.format_help())
    values = list(foot.sole_offset) + [getattr(foot, name) for name in (
        "velocity_cutoff_hz", "force_on_n", "force_off_n", "confirm_on_s", "confirm_off_s", "encoder_noise_std",
    )]
    if not all(math.isfinite(value) for value in values):
        extra.error("Foot-state options must be finite")
    if not 0 <= foot.force_off_n < foot.force_on_n or foot.velocity_cutoff_hz <= 0:
        extra.error("Require 0 <= force_off_n < force_on_n and positive cutoff")
    if min(foot.confirm_on_s, foot.confirm_off_s, foot.encoder_noise_std) < 0:
        extra.error("Confirmation times and encoder noise must be nonnegative")
    if (not all(math.isfinite(getattr(foot, key)) and getattr(foot, key) > 0 for key in reference_defaults)
            or foot.nominal_step_m > foot.max_step_m):
        extra.error("Reference options must be finite/positive, with nominal_step_m <= max_step_m")
    if not all(math.isfinite(gap) and gap >= 0.15 for gap in foot.curb_gaps):
        extra.error("--curb_gaps must be finite and >= 0.15 m")
    if "--curb_gap_range" in remaining and foot.curb_layout == "alternating":
        extra.error("Use --curb_layout sampled with --curb_gap_range, or use --curb_gaps")
    # A demo-specific default; explicit CLI count still wins, including hard runs.
    if not any(arg == "--curb_count" or arg.startswith("--curb_count=") for arg in remaining):
        remaining += ["--curb_count", "5"]
    if not any(flag in remaining for flag in ("--dry_run", "--help", "-h")):
        # cuDNN can re-exec: preserve the COMPLETE command line at that boundary.
        from render_bootstrap import preload_isaac_numpy
        from cudnn_bootstrap import preload_torch_cudnn

        preload_isaac_numpy()
        preload_torch_cudnn()
    original = sys.argv
    try:
        sys.argv = [original[0]] + remaining
        args, cases, launcher = evaluation.parse_args({
            "task": "Instinct-Parkour-Mixed-Amp-G1-v0", "terrain_mode": "mixed",
            "num_envs": 2, "num_steps": 4, "episodes_per_env": 1,
            "episode_length_s": 120.0, "max_steps": 6000,
        })
    finally:
        sys.argv = original
    if args.trace_stride != 1:
        extra.error("This demo logs every step; use --trace_stride 1")
    from foothold_reference import LaneGeometry, ReferenceConfig, arrange_curbs

    cases = arrange_curbs(cases, foot.curb_gaps if foot.curb_layout == "alternating" else None)
    reference = ReferenceConfig(**{name: getattr(foot, name) for name in ReferenceConfig.__dataclass_fields__})
    for case in cases:
        geometry = LaneGeometry(case, reference)
        case["curb_reference_rules"] = [geometry.curb_rule(i) for i in range(len(case.get("curbs", [])) - 1)]
    return args, foot, cases, launcher


def run_simulation(args, foot, cases, output, diagnostics, simulation_app):
    diagnostics.phase("import_demo_modules")
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import parse_env_cfg

    import instinctlab.tasks  # noqa: F401
    import cli_args
    from instinctlab.tasks.parkour.config.g1.stair_eval_cfg import configure_stair_evaluation
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    from demo_output import DemoOutput, overlay_frame
    from foot_state import FootStateObserver, LegKinematics, ObserverConfig, quaternion_matrix
    from foothold_output import FootholdOutput
    from foothold_reference import ReferenceConfig
    from sim_snapshot import FootSnapshotSuccess

    args.resume, args.run_name = True, None
    agent_cfg = cli_args.parse_instinct_rl_cfg(args.task, args)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    run_dir = Path(args.load_run).expanduser()
    if not run_dir.is_absolute():
        run_dir = Path("logs/instinct_rl") / agent_cfg.experiment_name / run_dir
    checkpoint = (run_dir / args.checkpoint).resolve(strict=True)
    env_path, agent_path = run_dir / "params/env.yaml", run_dir / "params/agent.yaml"
    restoration = None
    if args.use_current_cfg:
        agent = agent_cfg.to_dict()
    else:
        diagnostics.phase("restore_saved_training_config")
        restoration = evaluation.restore_training_env_config(
            cfg, evaluation.load_training_yaml(env_path), skip_paths=("/scene/terrain/terrain_generator",),
        )
        agent = evaluation.load_training_yaml(agent_path)
    cfg = configure_stair_evaluation(
        cfg, cases, args.seed, args.speed, args.episode_length_s, args.success_hold_s,
        command_mode=args.command_mode, centered_start=args.centered_start,
    )
    cfg.sim.device = args.device
    # Only this demo's config uses the sampler; its return is inherited unchanged.
    cfg.terminations.stair_success.func = FootSnapshotSuccess
    if args.video:
        spacing = cfg.scene.terrain.terrain_generator.size[1]
        lane_y = (args.video_env_id - (args.num_envs - 1) / 2) * spacing
        cfg.viewer.origin_type = "world"
        cfg.viewer.eye, cfg.viewer.lookat = (
            evaluation.video_overview_pose(cases, spacing) if args.video_view == "overview"
            else evaluation.video_camera_pose(cases[args.video_env_id], lane_y)
        )
        cfg.viewer.resolution = (args.video_width, args.video_height)
        import PIL  # noqa: F401 -- fail before rollout if annotation dependency is missing
    observer_config = ObserverConfig(
        dt=cfg.sim.dt * cfg.decimation, velocity_cutoff_hz=foot.velocity_cutoff_hz,
        force_on_n=foot.force_on_n, force_off_n=foot.force_off_n,
        confirm_on_s=foot.confirm_on_s, confirm_off_s=foot.confirm_off_s,
    )
    reference_config = ReferenceConfig(**{name: getattr(foot, name) for name in ReferenceConfig.__dataclass_fields__})
    agent["device"] = args.device
    dump_yaml(str(output / "eval_env.yaml"), cfg)
    dump_yaml(str(output / "eval_agent.yaml"), agent)
    manifest = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    urdf = Path(cfg.scene.robot.spawn.asset_path).resolve(strict=True)
    manifest.update({
        "checkpoint": str(checkpoint), "checkpoint_sha256": evaluation.sha256(checkpoint),
        "urdf": str(urdf), "urdf_sha256": evaluation.sha256(urdf),
        "saved_config_restore": restoration, "observer_config": vars(observer_config),
        "reference_config": vars(reference_config),
        "saved_env_sha256": evaluation.sha256(env_path) if env_path.is_file() else None,
        "saved_agent_sha256": evaluation.sha256(agent_path) if agent_path.is_file() else None,
        "measurement_scope": (
            "Observer: joints and simulated up-force. Separate passive references/edge diagnostics: privileged root poses and terrain geometry."
        ),
    })
    evaluation.write_json(output / "cases.json", manifest)
    log = DemoOutput(output)
    env, recorder, footholds = None, None, None
    counters = np.zeros(args.num_envs, dtype=int)
    outcomes = []
    steps, status, error_text, video_error = 0, "initializing", None, None
    report = {"mode": "isaac_sim_frozen_policy", "observer_config": vars(observer_config)}
    try:
        diagnostics.phase("create_environment")
        env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
        env = InstinctRlVecEnvWrapper(env)
        base = env.unwrapped
        evaluation.validate_policy_observations(base)
        kinematics = LegKinematics(urdf, list(base.scene["robot"].joint_names), sole_offset=foot.sole_offset)
        if kinematics.root_link != "torso_link":
            raise ValueError("Demo truth comparison requires the actual torso-root URDF")
        observer = FootStateObserver(args.num_envs, observer_config)
        # Follow actual terrain assignment, rather than assuming environment index == lane index.
        assigned_cases = base._stair_eval_cases
        footholds = FootholdOutput(output, assigned_cases, reference_config, foot.force_on_n)
        lane_origins = base.scene.env_origins[:, :2].detach().cpu().numpy().copy()
        rng = np.random.RandomState(args.seed + 771)
        diagnostics.phase("load_frozen_actor", str(checkpoint))
        policy, iteration = evaluation.build_inference_policy(env, agent, checkpoint, args.device, sample=args.sample)
        report.update({"checkpoint": str(checkpoint), "iteration": int(iteration),
                       "root_link": kinematics.root_link, "required_joint_names": kinematics.required_joint_names})
        observation, _ = env.get_observations()
        sampler = base.termination_manager.get_term_cfg("stair_success").func
        if args.video:
            diagnostics.phase("initialize_mp4")
            recorder = evaluation.Mp4Recorder(output / "foot_state.mp4", 1 / (base.step_dt * args.video_stride))
            recorder.append(evaluation.first_render_frame(base))
        diagnostics.phase("observe_frozen_policy")
        status = "running"
        with torch.inference_mode():
            while steps < args.max_steps and simulation_app.is_running():
                if not torch.isfinite(observation).all():
                    raise RuntimeError("Nonfinite actor observation")
                action = policy(observation)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Nonfinite actor action")
                observation, _, dones, _ = env.step(action)
                steps += 1
                raw = {key: value.detach().cpu().numpy() for key, value in sampler.foot_snapshot.items()}
                done = dones.bool().cpu().numpy()
                active = counters < args.episodes_per_env
                flags = {name: base.termination_manager.get_term(name).bool().cpu().numpy()
                         for name in base.termination_manager.active_terms}
                reasons = ["|".join(name for name, values in flags.items() if values[i]) for i in range(args.num_envs)]
                measured_q = raw["q"].copy()
                if foot.encoder_noise_std:
                    measured_q += rng.normal(0, foot.encoder_noise_std, measured_q.shape)
                started = time.perf_counter()
                position, orientation, velocity = kinematics.compute(measured_q, raw["dq"])
                state = observer.update(steps, steps * base.step_dt, position, orientation, velocity,
                                        np.maximum(0, raw["force_w"][..., 2]))
                log.update_ms.append((time.perf_counter() - started) * 1000)
                root_r = quaternion_matrix(raw["root_quaternion_w"])
                ankle_r = quaternion_matrix(raw["ankle_quaternion_w"])
                truth_w = raw["ankle_position_w"] + np.einsum("nfij,j->nfi", ankle_r, kinematics.sole_offset)
                truth_b = np.einsum("nji,nfj->nfi", root_r, truth_w - raw["root_position_w"][:, None])
                errors = np.linalg.norm(position - truth_b, axis=-1)
                log.append(steps, steps * base.step_dt, state, raw["force_w"], counters,
                           raw["episode_step"], done, reasons, active, errors=errors)
                position_lane = np.einsum("nij,nfj->nfi", root_r, position) + raw["root_position_w"][:, None]
                position_lane[..., :2] -= lane_origins[:, None]
                orientation_lane = root_r[:, None] @ orientation
                footholds.append(steps, steps * base.step_dt, state, position_lane, orientation_lane,
                                 raw["force_w"], counters, raw["episode_step"], active)
                for env_id in np.flatnonzero(done & active):
                    outcomes.append({
                        "env_id": int(env_id), "episode_index": int(counters[env_id]),
                        "outcome": evaluation.classify_episode(
                            {name: values[env_id] for name, values in flags.items()}
                        ),
                        "termination_reasons": reasons[env_id],
                    })
                    counters[env_id] += 1
                # The sampler retains the terminal frame; reset only our own state now.
                if (
                    recorder is not None and steps % args.video_stride == 0
                    and (args.video_length is None or steps <= args.video_length)
                ):
                    frame = overlay_frame(base.render(), state, args.video_env_id, steps,
                                          done=bool(done[args.video_env_id]))
                    recorder.append(footholds.overlay(frame, args.video_env_id))
                observer.reset(np.flatnonzero(done))
                footholds.reset(np.flatnonzero(done))
                if steps % 100 == 0:
                    print(
                        f"[FOOT] step={steps}, states={state['state'].tolist()}, max FK error={errors.max():.6g} m",
                        flush=True,
                    )
                if (counters >= args.episodes_per_env).all():
                    status = "complete"
                    break
            if status != "complete":
                status = "step_limit" if steps >= args.max_steps else "simulation_closed"
    except BaseException as error:
        status, error_text = "error", f"{type(error).__name__}: {error}"
        raise
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception as error:
                video_error = str(error)
                status, error_text = "error", error_text or video_error
        report.update({
            "status": status, "error": error_text, "simulation_steps": steps,
            "expected_episodes": args.num_envs * args.episodes_per_env,
            "completed_episodes": int(counters.sum()), "episodes": outcomes,
            "video": {"enabled": args.video, "path": str(recorder.path) if recorder else None,
                      "frames": recorder.frames if recorder else 0, "encoding_error": video_error},
        })
        try:
            if footholds is not None:
                report["foothold_diagnostics"] = footholds.close(status)
            log.close(report)
        finally:
            if env is not None:
                env.close()
        if video_error and sys.exc_info()[0] is None:
            raise RuntimeError(f"MP4 finalization failed: {video_error}")


def main():
    args, foot, cases, launcher = parse_args()
    output = evaluation.create_output_directory(
        args.output_dir or Path("outputs/foot_state_demo") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
    )
    evaluation.write_json(output / "cases.json", {
        "protocol": "foot_state_demo_v2", "cases": cases, "foot_options": vars(foot),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": ("Passive foot state, privileged frozen-world foothold targets and edge diagnostics. "
                  "The actor receives no foothold targets and no training reward is changed."),
    })
    print(f"[INFO] Independent demo output: {output}", flush=True)
    if args.dry_run:
        return
    with evaluation.StartupDiagnostics(output) as diagnostics:
        diagnostics.phase("launch_isaac_sim")
        app = launcher(args).app
        try:
            run_simulation(args, foot, cases, output, diagnostics, app)
        finally:
            app.close()


if __name__ == "__main__":
    main()
