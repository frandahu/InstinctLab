"""Reproducible staircase cases and result bookkeeping (no simulator imports)."""

from __future__ import annotations

import math
import random
from collections import Counter


def make_cases(
    num_envs: int,
    seed: int,
    mode: str,
    num_steps: int,
    height_range: tuple[float, float],
    depth_range: tuple[float, float],
    width: float,
    irregular: bool,
) -> list[dict]:
    """Each lane has its own seed; adding lanes preserves existing cases."""
    if num_envs < 1 or num_steps < 1:
        raise ValueError("num_envs and num_steps must be positive")
    if mode not in ("mixed", "up", "down"):
        raise ValueError("stair_mode must be mixed, up, or down")
    if mode == "mixed" and num_envs < 2:
        raise ValueError("mixed mode requires at least two environments")
    for name, bounds in (("step_height_range", height_range), ("tread_depth_range", depth_range)):
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f"{name} requires finite 0 < minimum <= maximum")
    if not math.isfinite(width) or width < 0.8:
        raise ValueError("stair_width must be finite and at least 0.8 m")
    cases = []
    for env_id in range(num_envs):
        case_seed = seed + 1009 * env_id
        rng = random.Random(case_seed)
        direction = ("up" if env_id % 2 == 0 else "down") if mode == "mixed" else mode
        heights = [rng.uniform(*height_range) for _ in range(num_steps if irregular else 1)]
        depths = [rng.uniform(*depth_range) for _ in range(num_steps if irregular else 1)]
        if not irregular:
            heights *= num_steps
            depths *= num_steps
        start_height = 0.0 if direction == "up" else sum(heights)
        surface_heights = []
        level = start_height
        for height in heights:
            level += height if direction == "up" else -height
            surface_heights.append(max(0.0, level))
        # Start at x=0; the first riser is 1.2 m ahead. All distances are lane-local.
        stair_start = 1.2
        stair_end = stair_start + sum(depths)
        cases.append({
            "env_id": env_id,
            "case_seed": case_seed,
            "direction": direction,
            "irregular": irregular,
            "riser_heights_m": heights,
            "tread_depths_m": depths,
            "surface_heights_m": surface_heights,
            "width_m": width,
            "start_height_m": start_height,
            "end_height_m": surface_heights[-1],
            "lane_min_x_m": -1.2,
            "stair_start_x_m": stair_start,
            "stair_end_x_m": stair_end,
            "goal_x_m": stair_end + 0.8,
            "lane_max_x_m": stair_end + 1.6,
        })
    return cases


def surface_height(case: dict, x: float) -> float:
    """Analytic surface height for diagnostics; never a policy observation."""
    if x < case["stair_start_x_m"]:
        return case["start_height_m"]
    edge = case["stair_start_x_m"]
    for depth, height in zip(case["tread_depths_m"], case["surface_heights_m"]):
        edge += depth
        if x < edge:
            return height
    return case["end_height_m"]


def classify_episode(flags: dict[str, bool]) -> str:
    """A simultaneous fall and goal crossing always counts as failure."""
    for name in ("nonfinite_state", "base_contact", "bad_orientation", "root_height", "off_course"):
        if flags.get(name, False):
            return name
    if flags.get("stair_success", False):
        return "success"
    if flags.get("time_out", False):
        return "timeout"
    return "other_termination"


def summarize(episodes: list[dict], expected: int) -> dict:
    counts = Counter(row["outcome"] for row in episodes)
    successful = [row for row in episodes if row["outcome"] == "success"]
    return {
        "expected_episodes": expected,
        "completed_episodes": len(episodes),
        "uncompleted_episodes": expected - len(episodes),
        "outcome_counts": dict(counts),
        "success_rate_completed": len(successful) / len(episodes) if episodes else None,
        "success_rate_planned": len(successful) / expected if expected else None,
        "mean_success_time_s": (
            sum(row["elapsed_s"] for row in successful) / len(successful) if successful else None
        ),
    }
