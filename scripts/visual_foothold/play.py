"""Record the B actor with MP4 and pre-reset commanded-touchdown diagnostics.

Uses saved training terrain/configuration. This is a training-distribution
playback, not the independent terrain-route success benchmark.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/instinct_rl"))
from cudnn_bootstrap import preload_torch_cudnn

preload_torch_cudnn()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--load_run", required=True, help="Absolute B training run directory")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default="outputs/visual_foothold/play")
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.max_steps) < 1:
    parser.error("num_envs and max_steps must be positive")
args.enable_cameras = args.enable_cameras or args.video
app = AppLauncher(args)


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.envs import ViewerCfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import parse_env_cfg
    import instinctlab.tasks  # noqa: F401
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
    from eval_stairs import build_inference_policy, create_output_directory, sha256
    from training_yaml import load_training_yaml, restore_training_env_config
    from mp4_video import Mp4Recorder, first_render_frame

    run = Path(args.load_run).expanduser().resolve(strict=True)
    checkpoint = (run / args.checkpoint).resolve(strict=True)
    task = "Instinct-Parkour-Visual-Foothold-Amp-G1-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    saved = load_training_yaml(run / "params/env.yaml")
    if "visual_foothold" not in saved:
        raise ValueError("This playback requires a B visual foothold run")
    restore_training_env_config(cfg, saved)
    agent = load_training_yaml(run / "params/agent.yaml")
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    cfg.scene.terrain.terrain_generator.seed = args.seed
    # Frozen actor needs no discriminator or reference dataset. Actor/critic
    # observation terms and action scaling retain their saved dimensions.
    cfg.scene.motion_reference = None
    cfg.observations.amp_policy = None
    cfg.observations.amp_reference = None
    cfg.terminations.dataset_exhausted = None
    cfg.viewer = ViewerCfg(eye=(2.5, 2.5, 1.8), lookat=(0., 0., 0.7),
                           origin_type="asset_root", asset_name="robot", resolution=(960, 540))
    output = create_output_directory(Path(args.output_dir))
    dump_yaml(str(output / "eval_env.yaml"), cfg)
    dump_yaml(str(output / "eval_agent.yaml"), agent)
    env, recorder = None, None
    errors, valid_ticks, rows, resets = [], 0, 0, 0
    result = {"status": "running", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "seed": args.seed, "action_mode": "deterministic_mean",
              "scope": "Training-terrain playback; depth/FK diagnostics, not route success or stability certification"}
    try:
        env = InstinctRlVecEnvWrapper(gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None))
        base = env.unwrapped
        base.visual_recording = True
        policy, iteration = build_inference_policy(env, agent, checkpoint, args.device)
        result["iteration"] = iteration
        if args.video:
            from PIL import Image, ImageDraw
            first_render_frame(base)
            recorder = Mp4Recorder(output / "visual_foothold.mp4", fps=1 / (2*base.step_dt))
        obs, _ = env.get_observations()
        episode = np.zeros(args.num_envs, dtype=int)
        with (output / "visual_trace.csv").open("x", newline="", encoding="utf-8") as stream:
            fields = ["global_step", "env_id", "episode", "time_s", "swing_foot", "target_valid", "anchor_valid",
                      "target_x_b_m", "target_y_b_m", "target_z_b_m", "left_state", "right_state",
                      "commanded_touchdown", "touchdown_error_m", "done"]
            writer = csv.writer(stream)
            writer.writerow(fields)
            with torch.inference_mode():
                for tick in range(args.max_steps):
                    if not app.app.is_running():
                        raise RuntimeError("Simulation application closed before max_steps")
                    if not torch.isfinite(obs).all():
                        raise RuntimeError("Nonfinite policy observation")
                    action = policy(obs)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite policy action")
                    obs, _, dones, _ = env.step(action)
                    snapshot = base.visual_snapshot
                    values = snapshot["observation"].cpu().numpy()
                    landed = snapshot["touchdown"].cpu().numpy()
                    error = snapshot["error"].cpu().numpy()
                    times = snapshot["time"].cpu().numpy()
                    contact = snapshot["contact_state"].cpu().numpy()
                    done = dones.cpu().numpy().astype(bool)
                    errors.extend(error[landed].tolist())
                    valid_ticks += int((values[:, 32] > 0.5).sum())
                    for e in range(args.num_envs):
                        writer.writerow([tick+1, e, episode[e], times[e], int(values[e, 29] > 0.5),
                                         int(values[e, 32] > 0.5), int(values[e, 33] > 0.5), *values[e, 22:25],
                                         *contact[e], int(landed[e]), float(error[e]) if landed[e] else "", int(done[e])])
                    if recorder is not None and tick % 2 == 0:
                        frame = Image.fromarray(base.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, frame.width, 64), fill="black")
                        draw.text((10, 8), f"B: DEPTH foothold | env0 target valid={int(values[0,32])} "
                                  f"swing={'right' if values[0,29] else 'left'}", fill="white")
                        draw.text((10, 30), f"target torso XYZ m: {values[0,22]:+.2f}, {values[0,23]:+.2f}, "
                                  f"{values[0,24]:+.2f} | measured stance proxy", fill="white")
                        recorder.append(np.asarray(frame))
                    episode += done
                    resets += int(done.sum())
                    rows += args.num_envs
        result["status"] = "complete"
    except Exception as error:
        result["status"], result["error"] = "error", str(error)
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
                result["video_frames"] = recorder.frames
        except Exception as error:
            result["status"], result["encoding_error"] = "error", str(error)
            raise
        finally:
            result.update(trace_rows=rows, resets=resets, commanded_touchdowns=len(errors),
                          valid_target_fraction=valid_ticks / max(rows, 1),
                          touchdown_error_mean_m=float(np.mean(errors)) if errors else None,
                          touchdown_error_p95_m=float(np.percentile(errors, 95)) if errors else None)
            (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if env is not None:
                env.close()
        print(f"[VISUAL] Playback artifacts: {output}", flush=True)


try:
    main()
finally:
    app.app.close()
