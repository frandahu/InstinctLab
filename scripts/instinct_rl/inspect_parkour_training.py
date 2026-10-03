"""Print saved Parkour settings and relevant TensorBoard metrics without Isaac Sim."""

import argparse
import sys
import types
from pathlib import Path

from training_yaml import load_training_yaml


METRIC_NAMES = (
    "track_lin_vel_xy_exp", "tracking_exp_vel_xy", "error_vel_xy",
    "dont_wait", "feet_air_time", "is_alive", "mean_noise_std", "mean_episode_length",
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
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.at_iteration is not None and args.at_iteration < 0:
        parser.error("--at_iteration must be nonnegative")
    run_dir = args.load_run.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"Training run directory does not exist: {run_dir}")
    print(f"[RUN] {run_dir}")
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
