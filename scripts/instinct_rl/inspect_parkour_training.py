"""Print saved Parkour settings and relevant TensorBoard metrics without Isaac Sim."""

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path

from training_yaml import load_training_yaml


METRIC_NAMES = (
    "track_lin_vel_xy_exp", "tracking_exp_vel_xy", "error_vel_xy",
    "dont_wait", "feet_air_time", "is_alive", "mean_noise_std", "mean_episode_length",
    "route_progress", "terrain_level", "action_rate", "freeze_upper_body",
    "stair_goal",
    "tracking_exp_vel_yaw", "error_vel_yaw", "heading_error", "feet_slide", "discriminator",
)


def read_training_metrics(run_dir, samples=3, through_iteration=None):
    # TensorBoard's supported no-TensorFlow compatibility path keeps this
    # standalone reader from importing an installed TensorFlow GPU runtime.
    sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0}, purge_orphaned_data=False)
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    result = {}
    for tag in sorted(tags):
        # Runner tags under /time/ use elapsed wall time as their event step.
        # A checkpoint iteration cutoff must never be applied to that axis.
        if "time" in tag.lower().split("/"):
            continue
        if not any(name in tag.lower() for name in METRIC_NAMES):
            continue
        # Keep the most recently written value if a restarted run reused a step.
        by_step = {}
        for event in sorted(accumulator.Scalars(tag), key=lambda item: item.wall_time):
            if through_iteration is None or event.step <= through_iteration:
                by_step[event.step] = event.value
        rows = sorted(by_step.items())[-samples:]
        if rows:
            result[tag] = rows
    return result, sorted(tags)


def saved_reward_settings(run_dir):
    path = Path(run_dir) / "params/env.yaml"
    if not path.is_file():
        return {}
    config = load_training_yaml(path)
    groups = config.get("rewards", {}) or {}
    result = {}
    for group_name, group in groups.items():
        if not isinstance(group, dict):
            continue
        for name in ("track_lin_vel_xy_exp", "dont_wait", "feet_air_time", "is_alive"):
            term = group.get(name)
            if isinstance(term, dict):
                result[f"{group_name}/{name}"] = {
                    "weight": term.get("weight"), "std": (term.get("params", {}) or {}).get("std"),
                }
    return result


def saved_training_setup(run_dir):
    """Read what was actually saved, without inferring settings from run names."""
    run_dir = Path(run_dir)
    env_path, agent_path = run_dir / "params/env.yaml", run_dir / "params/agent.yaml"
    env = load_training_yaml(env_path) if env_path.is_file() else {}
    agent = load_training_yaml(agent_path) if agent_path.is_file() else {}
    count = (env.get("scene", {}) or {}).get("num_envs")
    steps = agent.get("num_steps_per_env")
    algorithm = agent.get("algorithm", {}) or {}
    minibatches = algorithm.get("num_mini_batches")
    rollout = count * steps if type(count) is int and type(steps) is int else None
    inventory_path = run_dir / "params/motion_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    from prepare_parkour_references import FILES

    motions = inventory.get("motions", [])
    author_hash = FILES["parkour_motion_without_run_retargetted.npz"][1]
    reference = (env.get("scene", {}).get("motion_reference", {}) or {}).get("motion_buffers", {}).get("run_walk", {}) or {}
    return {
        "num_envs": count, "steps_per_env": steps, "transitions_per_iteration": rollout,
        "transitions_per_minibatch": rollout // minibatches if rollout and type(minibatches) is int and minibatches > 0 else None,
        "configured_max_iterations": agent.get("max_iterations"), "resume": agent.get("resume"),
        "algorithm": algorithm.get("class_name"), "AMP_coef": algorithm.get("discriminator_reward_coef"),
        "motion_root": inventory.get("root", reference.get("path")),
        "selection": inventory.get("selection", reference.get("filtered_motion_selection_filepath")),
        "motion_files": inventory.get("files"), "motion_frames": inventory.get("frames"),
        "author_motion_hash_matches": len(motions) == 1 and motions[0].get("sha256") == author_hash,
    }


def read_evaluation_metrics(eval_dir):
    """Summarize completed or interrupted traces, per lane and episode.

    Body-frame speed error and world-frame path drift are distinct quantities.
    Ignore the first second and near-zero forward commands for speed averages.
    Older traces have no absolute heading; do not infer it from yaw rate.
    """
    groups = {}
    path = Path(eval_dir) / "trace.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["env_id"]), int(row["episode_index"]))
            fields = ("elapsed_s", "base_x_m", "base_y_m", "vel_x_m_s", "vel_y_m_s",
                      "command_x_m_s", "command_y_m_s", "command_yaw_rad_s")
            values = {name: float(row[name]) for name in fields}
            if not all(math.isfinite(value) for value in values.values()):
                continue
            heading = row.get("heading_w_rad")
            values["heading"] = float(heading) if heading else None
            groups.setdefault(key, []).append(values)
    result = []
    for (env_id, episode), rows in sorted(groups.items()):
        first, last = rows[0], rows[-1]
        dx, dy = last["base_x_m"] - first["base_x_m"], last["base_y_m"] - first["base_y_m"]
        moving = [r for r in rows if r["elapsed_s"] >= 1.0 and r["command_x_m_s"] > .15]
        def mean(name):
            return sum(r[name] for r in moving) / len(moving) if moving else None
        headings = [abs(math.atan2(math.sin(r["heading"]), math.cos(r["heading"])))
                    for r in moving if r["heading"] is not None and math.isfinite(r["heading"])]
        result.append({
            "env_id": env_id, "episode": episode, "trace_duration_s": last["elapsed_s"],
            "delta_x_m": dx, "delta_y_m": dy,
            "net_path_angle_deg": math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > .05 else None,
            "max_abs_lateral_offset_m": max(abs(r["base_y_m"]) for r in rows),
            "moving_samples": len(moving), "mean_command_x_m_s": mean("command_x_m_s"),
            "mean_forward_speed_body_m_s": mean("vel_x_m_s"),
            "mean_lateral_speed_body_m_s": mean("vel_y_m_s"),
            "mean_velocity_error_body_m_s": sum(math.hypot(r["vel_x_m_s"] - r["command_x_m_s"],
                                                          r["vel_y_m_s"] - r["command_y_m_s"]) for r in moving) / len(moving) if moving else None,
            "mean_abs_command_yaw_rad_s": sum(abs(r["command_yaw_rad_s"]) for r in moving) / len(moving) if moving else None,
            "mean_abs_heading_world_deg": math.degrees(sum(headings) / len(headings)) if headings else None,
        })
    return result


def inspect_checkpoint(path):
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    std = state.get("std")
    stats = None
    if std is not None:
        stats = (std.float().mean().item(), std.min().item(), std.max().item())
    return checkpoint.get("iter"), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_run", required=True, type=Path, help="Exact training run directory")
    parser.add_argument("--checkpoint", help="Optional exact filename; also selects metrics up to its saved iteration")
    parser.add_argument("--samples", type=int, default=3, help="Recent values to show for each relevant metric")
    parser.add_argument("--at_iteration", type=int, help="Override the metric iteration cutoff")
    parser.add_argument("--eval_dir", type=Path, help="Also summarize this evaluation's trace.csv (no Isaac Sim required)")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.at_iteration is not None and args.at_iteration < 0:
        parser.error("--at_iteration must be nonnegative")
    run_dir = args.load_run.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"Training run directory does not exist: {run_dir}")
    print(f"[RUN] {run_dir}")
    print("[TRAINING_SETUP] " + json.dumps(saved_training_setup(run_dir), ensure_ascii=False))
    if args.eval_dir:
        for row in read_evaluation_metrics(args.eval_dir):
            print("[FLAT_TRACE] " + json.dumps(row, ensure_ascii=False))
    cutoff = args.at_iteration
    if args.checkpoint:
        if Path(args.checkpoint).name != args.checkpoint or args.checkpoint in (".", ".."):
            parser.error("--checkpoint must be a filename inside --load_run")
        path = run_dir / args.checkpoint
        if not path.is_file():
            parser.error(f"Checkpoint does not exist: {path}")
        iteration, std = inspect_checkpoint(path)
        print(f"[CHECKPOINT] file={path.name}, iteration={iteration}")
        if std is not None:
            print(f"[ACTION_STD] mean={std[0]:.4f}, min={std[1]:.4f}, max={std[2]:.4f}")
        if cutoff is None:
            cutoff = iteration
    for name, settings in saved_reward_settings(run_dir).items():
        print(f"[SAVED_CONFIG] {name}: weight={settings['weight']}, std={settings['std']}")
    # Locate event files explicitly so a missing log is reported clearly.
    if not any(run_dir.glob("events.out.tfevents.*")):
        print("[WARN] No TensorBoard event files in this directory. Saved settings/checkpoint alone cannot show learning quality.")
        return
    metrics, tags = read_training_metrics(run_dir, args.samples, cutoff)
    print(f"[METRICS] iteration <= {cutoff if cutoff is not None else 'latest'}, last {args.samples} values per tag")
    for tag, rows in metrics.items():
        print(f"[METRIC] {tag}: " + "; ".join(f"iter={step}, value={value:.6g}" for step, value in rows))
    if not metrics:
        print("[WARN] No matching scalar values at the requested iteration. Available scalar tags:")
        for tag in tags[:30]:
            print(f"  {tag}")
    print("[NOTE] Reward tags include their saved weights and may use different time normalizations; values are not physical velocities.")


if __name__ == "__main__":
    main()
