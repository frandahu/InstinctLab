"""Reproducible staircase cases and result bookkeeping (no simulator imports)."""

from __future__ import annotations

import copy
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
    irregular_fraction: float = 0.20,
    dimension_variation: float = 0.25,
    landing_depth: float = 1.20,
    vary_both_dimensions: bool = False,
) -> list[dict]:
    """Each lane has its own seed; adding lanes preserves existing cases."""
    if num_envs < 1 or num_steps < 1:
        raise ValueError("num_envs and num_steps must be positive")
    if mode not in ("up_down", "mixed", "up", "down"):
        raise ValueError("stair_mode must be up_down, mixed, up, or down")
    if mode == "mixed" and num_envs < 2:
        raise ValueError("mixed mode requires at least two environments")
    for name, bounds in (("step_height_range", height_range), ("tread_depth_range", depth_range)):
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f"{name} requires finite 0 < minimum <= maximum")
    if not math.isfinite(width) or width < 0.8:
        raise ValueError("stair_width must be finite and at least 0.8 m")
    if not math.isfinite(landing_depth) or landing_depth < 0.8:
        raise ValueError("landing_depth must be finite and at least 0.8 m")
    if not math.isfinite(irregular_fraction) or not 0 < irregular_fraction < 0.5:
        raise ValueError("irregular_fraction must be finite and in (0, 0.5)")
    if not math.isfinite(dimension_variation) or not 0 < dimension_variation < 1:
        raise ValueError("dimension_variation must be finite and in (0, 1)")
    if irregular and num_steps < 3:
        raise ValueError("Sparse irregular stairs require at least three steps per flight")
    if irregular and height_range[0] == height_range[1] and depth_range[0] == depth_range[1]:
        raise ValueError("Irregular stairs require a nonzero height or depth range; use --regular otherwise")
    if irregular and vary_both_dimensions and (height_range[0] == height_range[1] or depth_range[0] == depth_range[1]):
        raise ValueError("vary_both_dimensions requires nonzero height AND depth ranges")
    anomaly_count = min(math.ceil(num_steps * irregular_fraction), (num_steps - 1) // 2) if irregular else 0

    def vary(rng, baseline, bounds):
        # Keep dimensions within the requested bounds; never alter a fixed dimension.
        signs = [sign for sign, room in ((-1, baseline - bounds[0]), (1, bounds[1] - baseline)) if room > 0]
        sign = rng.choice(signs)
        room = baseline - bounds[0] if sign < 0 else bounds[1] - baseline
        delta = min(baseline * rng.uniform(dimension_variation / 2, dimension_variation), room)
        return baseline + sign * delta

    def flight(rng, direction, nominal_height, nominal_depth, carried_heights=None):
        heights, depths = [nominal_height] * num_steps, [nominal_depth] * num_steps
        indices = rng.sample(range(num_steps), anomaly_count)
        height_can_vary = height_range[0] < height_range[1]
        depth_can_vary = depth_range[0] < depth_range[1]
        if carried_heights is not None:
            # Reposition the ascent's height deviations on descent. Equal total rise
            # and fall brings the final platform back to ground level exactly.
            values = [height for height in carried_heights if height != nominal_height]
            rng.shuffle(values)
            for index, height in zip(indices, values):
                heights[index] = height
        for index in indices:
            if carried_heights is None:
                choices = (["height"] if height_can_vary else []) + (["depth"] if depth_can_vary else [])
                if height_can_vary and depth_can_vary:
                    choices.append("both")
                changed = "both" if vary_both_dimensions else rng.choice(choices)
                if changed in ("height", "both"):
                    heights[index] = vary(rng, nominal_height, height_range)
                if changed in ("depth", "both"):
                    depths[index] = vary(rng, nominal_depth, depth_range)
            elif vary_both_dimensions or heights[index] == nominal_height or (depth_can_vary and rng.choice((False, True))):
                depths[index] = vary(rng, nominal_depth, depth_range)
        return {
            "direction": direction,
            "riser_heights_m": heights,
            "tread_depths_m": depths,
            "anomalous_step_indices": sorted(indices),  # zero-based, within this flight
        }

    cases = []
    for env_id in range(num_envs):
        case_seed = seed + 1009 * env_id
        rng = random.Random(case_seed)
        direction = ("up" if env_id % 2 == 0 else "down") if mode == "mixed" else mode
        nominal_height, nominal_depth = rng.uniform(*height_range), rng.uniform(*depth_range)
        first = flight(rng, "up" if direction == "up_down" else direction, nominal_height, nominal_depth)
        flights = [first]
        heights, depths = first["riser_heights_m"], first["tread_depths_m"]
        start_height = sum(heights) if direction == "down" else 0.0
        deltas = [height if direction != "down" else -height for height in heights]
        stair_start = 1.2
        summit_start = summit_end = summit_height = 0.0
        if direction == "up_down":
            second = flight(rng, "down", nominal_height, nominal_depth, carried_heights=heights)
            flights.append(second)
            summit_height = sum(heights)
            summit_start = stair_start + sum(depths)
            summit_end = summit_start + landing_depth
            deltas = heights + [0.0] + [-height for height in second["riser_heights_m"]]
            heights = heights + [0.0] + second["riser_heights_m"]
            depths = depths + [landing_depth] + second["tread_depths_m"]
        surface_heights = []
        level = start_height
        for delta in deltas:
            level += delta
            surface_heights.append(max(0.0, level))
        if direction in ("down", "up_down"):
            surface_heights[-1] = 0.0  # avoid floating-point residue on the final ground platform
        # Start at x=0; the first riser is 1.2 m ahead. All distances are lane-local.
        stair_end = stair_start + sum(depths)
        cases.append({
            "env_id": env_id,
            "case_seed": case_seed,
            "direction": direction,
            "irregular": irregular,
            "nominal_riser_height_m": nominal_height,
            "nominal_tread_depth_m": nominal_depth,
            "anomalous_steps_per_flight": anomaly_count,
            "flights": flights,
            "riser_heights_m": heights,
            "riser_deltas_m": deltas,
            "tread_depths_m": depths,
            "surface_heights_m": surface_heights,
            "width_m": width,
            "start_height_m": start_height,
            "end_height_m": surface_heights[-1],
            "requires_summit": direction == "up_down",
            "summit_start_x_m": summit_start,
            "summit_end_x_m": summit_end,
            "summit_height_m": summit_height,
            "lane_min_x_m": -1.2,
            "stair_start_x_m": stair_start,
            "stair_end_x_m": stair_end,
            "goal_x_m": stair_end + 0.8,
            "lane_max_x_m": stair_end + 1.6,
        })
    return cases


def make_flat_control_cases(stair_cases: list[dict]) -> list[dict]:
    """Flatten the same routes to test locomotion without changing goals or seeds."""
    result = copy.deepcopy(stair_cases)
    for case in result:
        case["control_for_direction"] = case["direction"]
        case["direction"] = "flat"
        case["irregular"] = False
        case["nominal_riser_height_m"] = 0.0
        case["anomalous_steps_per_flight"] = 0
        case["flights"] = []
        for key in ("riser_heights_m", "riser_deltas_m", "surface_heights_m"):
            case[key] = [0.0] * len(case[key])
        for key in ("start_height_m", "end_height_m", "summit_height_m", "summit_start_x_m", "summit_end_x_m"):
            case[key] = 0.0
        case["requires_summit"] = False
    return result


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
        "summit_reached_episodes": sum(bool(row.get("summit_reached", False)) for row in episodes),
    }
