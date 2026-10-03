"""Seeded training geometry; intentionally independent of simulator imports."""

import math
import random


def training_cases(seed, num_rows, num_cols, num_steps=6):
    """Row zero is flat; later rows rise from 3 cm to 18 cm per step.

    Each non-flat route ascends, crosses a landing and descends to ground.
    Sparse tread/riser deviations start at row three. Training and evaluation
    use different generators and seeds; these routes are not held-out tests.
    """
    if num_rows < 3 or num_cols < 1 or num_steps < 3:
        raise ValueError("Need >=3 difficulty rows, >=1 column and >=3 steps")
    grid = []
    for row in range(num_rows):
        lanes = []
        for col in range(num_cols):
            rng = random.Random(seed + 1009 * col + 65537 * row)
            height = 0.0 if row == 0 else 0.03 + 0.15 * (row - 1) / (num_rows - 2)
            depth = rng.uniform(0.28, 0.40)
            up, down = [height] * num_steps, [height] * num_steps
            up_depth, down_depth = [depth] * num_steps, [depth] * num_steps
            anomaly_count = min(math.ceil(0.20 * num_steps), (num_steps - 1) // 2) if row >= 3 else 0
            up_ids = rng.sample(range(num_steps), anomaly_count)
            down_ids = rng.sample(range(num_steps), anomaly_count)
            for i, j in zip(up_ids, down_ids):
                # Matched total rise/fall, independently placed deviations.
                delta = height * rng.choice((-1, 1)) * rng.uniform(0.10, 0.20)
                up[i] += delta
                down[j] += delta
                up_depth[i] *= rng.uniform(0.80, 1.20)
                down_depth[j] *= rng.uniform(0.80, 1.20)
            deltas = up + [0.0] + [-x for x in down]
            depths = up_depth + [1.20] + down_depth
            level, surfaces = 0.0, []
            for delta in deltas:
                level += delta
                surfaces.append(max(0.0, level))
            surfaces[-1] = 0.0
            end = 1.20 + sum(depths)
            lanes.append(dict(
                env_id=col, direction="flat" if row == 0 else "up_down",
                start_height_m=0.0, end_height_m=0.0, width_m=2.0,
                stair_start_x_m=1.20, stair_end_x_m=end,
                lane_min_x_m=-1.20, lane_max_x_m=end + 1.60, goal_x_m=end + 0.80,
                tread_depths_m=depths, surface_heights_m=surfaces,
                riser_deltas_m=deltas, nominal_riser_height_m=height,
                anomalous_steps_per_flight=anomaly_count,
                ascent_anomalies=up_ids, descent_anomalies=down_ids,
            ))
        grid.append(lanes)
    return grid
