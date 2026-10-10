"""Seeded curb crossing curriculum, without simulator dependencies."""

import math
import random


def curb_crossing_cases(seed, num_rows, num_cols, gap_range=(0.18, 0.30), max_step_m=0.60):
    """Build three short crossings and one ground transition per curb lane.

    Every fourth column is flat to retain basic walking. The remaining columns
    advance from 18 cm to 30 cm gaps. Reachability is an engineering envelope,
    not a dynamics guarantee. The sole length/margin and step width match the
    foot-state demo; a 30 cm gap needs a 57.2 cm horizontal support-to-target step.
    """
    if num_rows < 2 or num_cols < 4:
        raise ValueError("Curb curriculum needs >=2 rows and >=4 columns")
    if not (0.15 <= gap_range[0] <= gap_range[1] and math.isfinite(max_step_m)):
        raise ValueError("Require finite increasing gaps >=0.15 m")
    if math.hypot(gap_range[1] + 0.236, 0.20) > max_step_m:
        raise ValueError("Largest curb gap exceeds the configured safe-center step envelope")
    grid = []
    for row in range(num_rows):
        lanes = []
        difficulty = row / (num_rows - 1)
        upper_gap = gap_range[0] + difficulty * (gap_range[1] - gap_range[0])
        for col in range(num_cols):
            rng = random.Random(seed + 1009 * col + 65537 * row)
            flat = col % 4 == 0
            x, depths, heights, curbs, required = 1.20, [], [], [], []
            for i in range(5):
                depth = rng.uniform(0.80, 1.10)
                height = 0.0 if flat else rng.uniform(0.12, 0.14 + 0.06 * difficulty)
                short = i != 1 and i < 4
                gap = rng.uniform(max(gap_range[0], upper_gap - 0.02), upper_gap) if short else 0.85
                if i == 4:
                    gap = 1.0
                curbs.append(dict(start_x_m=x, end_x_m=x + depth, height_m=height, depth_m=depth, gap_after_m=gap))
                depths.extend((depth, gap))
                heights.extend((height, 0.0))
                if i < 4:
                    required.append(short and not flat)
                x += depth + gap
            lanes.append(
                dict(
                    env_id=col,
                    terrain="flat" if flat else "curb",
                    direction="flat" if flat else "up_down",
                    start_height_m=0.0,
                    end_height_m=0.0,
                    width_m=2.0,
                    stair_start_x_m=1.20,
                    stair_end_x_m=x,
                    lane_min_x_m=-1.20,
                    lane_max_x_m=x + 1.60,
                    goal_x_m=x + 0.80,
                    tread_depths_m=depths,
                    surface_heights_m=heights,
                    curbs=curbs,
                    bridge_required=required,
                )
            )
        grid.append(lanes)
    return grid
