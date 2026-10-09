"""Seeded stairs, ramps and curbs for frozen-policy evaluation; standard library only."""

from __future__ import annotations

import math
import random

from stair_eval_cases import make_cases


TERRAIN_SUITE = ("flat", "stairs", "slope", "curb", "mixed")


def profile_mesh(segments, width):
    """Closed triangular prisms with an exact linear top, including vertical risers."""
    vertices, faces = [], []
    triangles = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    for segment in segments:
        left, right, low, high = segment
        bottom = min(-0.1, low - 0.1, high - 0.1)
        index = len(vertices)
        vertices.extend([
            [left, -width / 2, bottom], [right, -width / 2, bottom],
            [right, width / 2, bottom], [left, width / 2, bottom],
            [left, -width / 2, low], [right, -width / 2, high],
            [right, width / 2, high], [left, width / 2, low],
        ])
        faces.extend([[index + i for i in triangle] for triangle in triangles])
    return vertices, faces


def profile_height(case, x):
    """Same piecewise-linear geometry as the collision mesh, in lane coordinates."""
    for left, right, low, high in case["profile"]:
        if x < right:
            return low + (high - low) * max(0.0, x - left) / (right - left)
    return case["end_height_m"]


def make_terrain_cases(num_envs, seed, terrain_mode, num_steps, height_range, depth_range,
                       width, irregular, irregular_fraction, dimension_variation, landing_depth,
                       slope_angle_range=(5.0, 20.0), ramp_length=2.0,
                       curb_height_range=(0.08, 0.20), curb_count=1, curb_depth_range=None,
                       curb_gap_range=(1.0, 1.0), vary_both_dimensions=False):
    """A mixed route crosses stairs, a ramp pair, then curbs before the final goal.

    Profiles use [x_start, x_end, z_start, z_end]. Mandatory flat checkpoints at
    each obstacle summit require supported landings rather than airborne passage.
    """
    if terrain_mode not in (*TERRAIN_SUITE, "suite"):
        raise ValueError("Unknown terrain mode: " + terrain_mode)
    if terrain_mode == "suite" and num_envs < len(TERRAIN_SUITE):
        raise ValueError("suite requires at least 5 environments (one per terrain)")
    for name, bounds, upper in (("slope_angle_range", slope_angle_range, 45.0),
                                ("curb_height_range", curb_height_range, 0.5)):
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 < bounds[0] <= bounds[1] <= upper:
            raise ValueError(f"{name} requires finite 0 < MIN <= MAX <= {upper}")
    if not math.isfinite(ramp_length) or ramp_length < 0.8:
        raise ValueError("ramp_length must be finite and at least 0.8 m")
    if not isinstance(curb_count, int) or isinstance(curb_count, bool) or not 1 <= curb_count <= 20:
        raise ValueError("curb_count must be an integer in [1, 20]")
    curb_depth_range = (landing_depth, landing_depth) if curb_depth_range is None else curb_depth_range
    for name, bounds, minimum in (("curb_depth_range", curb_depth_range, 0.8),
                                  ("curb_gap_range", curb_gap_range, 0.15)):
        if (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                or not minimum <= bounds[0] <= bounds[1]):
            raise ValueError(f"{name} requires finite {minimum} <= MIN <= MAX")
    # Reuse the existing validation and sparse irregular stair sampler.
    stairs = make_cases(num_envs, seed, "up_down", num_steps, height_range, depth_range,
                        width, irregular, irregular_fraction, dimension_variation, landing_depth, vary_both_dimensions)
    result = []
    for original in stairs:
        case = dict(original)
        env_id = case["env_id"]
        terrain = TERRAIN_SUITE[env_id % len(TERRAIN_SUITE)] if terrain_mode == "suite" else terrain_mode
        rng = random.Random(case["case_seed"] + 7919)
        profile = [[-1.2, 1.2, 0.0, 0.0]]
        checkpoints = []
        edge = 1.2

        def span(length, low, high):
            nonlocal edge
            profile.append([edge, edge + length, low, high])
            edge += length

        def landing(height, label, depth=None):
            depth = landing_depth if depth is None else depth
            checkpoints.append({"start_x_m": edge, "end_x_m": edge + depth,
                                "height_m": height, "terrain": label})
            span(depth, height, height)

        if terrain in ("stairs", "mixed"):
            for depth, height in zip(original["tread_depths_m"], original["surface_heights_m"]):
                if abs(edge - original["summit_start_x_m"]) < 1e-8:
                    checkpoints.append({"start_x_m": edge, "end_x_m": edge + depth,
                                        "height_m": height, "terrain": "stairs"})
                span(depth, height, height)
            span(1.0, 0.0, 0.0)
        angle = rng.uniform(*slope_angle_range)
        if terrain in ("slope", "mixed"):
            rise = ramp_length * math.tan(math.radians(angle))
            span(ramp_length, 0.0, rise)
            landing(rise, "slope")
            span(ramp_length, rise, 0.0)
            span(1.0, 0.0, 0.0)
        curb = rng.uniform(*curb_height_range)
        curbs = []
        if terrain in ("curb", "mixed"):
            for index in range(curb_count):
                height = curb if index == 0 else rng.uniform(*curb_height_range)
                # Fixed dimensions do not consume RNG, preserving the baseline geometry.
                depth = (curb_depth_range[0] if curb_depth_range[0] == curb_depth_range[1]
                         else rng.uniform(*curb_depth_range))
                gap = (curb_gap_range[0] if curb_gap_range[0] == curb_gap_range[1]
                       else rng.uniform(*curb_gap_range))
                curbs.append({"start_x_m": edge, "end_x_m": edge + depth,
                              "height_m": height, "depth_m": depth, "gap_after_m": gap})
                landing(height, "curb", depth)  # abrupt rise at entry and drop on exit
                span(gap, 0.0, 0.0)
        if terrain == "flat":
            span(6.0, 0.0, 0.0)
        obstacle_end = edge
        span(1.6, 0.0, 0.0)
        # A flat control has one slab, without artificial coplanar collision seams.
        if terrain == "flat":
            profile = [[-1.2, edge, 0.0, 0.0]]
        summit = checkpoints[0] if checkpoints else {"start_x_m": 0.0, "end_x_m": 0.0, "height_m": 0.0}
        vertices, faces = profile_mesh(profile, width)
        case.update({
            "terrain": terrain, "direction": "flat" if terrain == "flat" else "up_down",
            "profile": profile, "mesh_vertices": vertices, "mesh_faces": faces,
            "checkpoints": checkpoints, "slope_angle_deg": angle if terrain in ("slope", "mixed") else None,
            "ramp_length_m": ramp_length if terrain in ("slope", "mixed") else None,
            "curb_height_m": curb if terrain in ("curb", "mixed") else None,
            "curbs": curbs, "curb_count": len(curbs),
            "start_height_m": 0.0, "end_height_m": 0.0,
            "requires_summit": bool(checkpoints),
            "summit_start_x_m": summit["start_x_m"], "summit_end_x_m": summit["end_x_m"],
            "summit_height_m": summit["height_m"], "stair_end_x_m": obstacle_end,
            "goal_x_m": obstacle_end + 0.8, "lane_max_x_m": edge,
            "surface_heights_m": [max(low, high) for _, _, low, high in profile],
        })
        # Stair-specific dimensions are meaningful only when stairs are present.
        if terrain not in ("stairs", "mixed"):
            for key in ("flights", "riser_heights_m", "riser_deltas_m", "tread_depths_m"):
                case[key] = []
            case["irregular"] = False
            case["anomalous_steps_per_flight"] = 0
        result.append(case)
    return result
