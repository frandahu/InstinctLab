"""Privileged geometric references for passive evaluation, without simulator imports.

Coordinates are fixed lane/world axes, not a moving support-foot frame. References
are advisory: the existing velocity-conditioned actor does not receive them.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np


def clip_footprint(polygon, left, right, bottom, top):
    """Clip an ordered projected sole polygon against a rectangular top surface."""
    points = [np.array(p, dtype=float) for p in polygon]
    for axis, bound, sign in ((0, left, 1), (0, right, -1), (1, bottom, 1), (1, top, -1)):
        if not points:
            break
        clipped = []
        previous = points[-1]
        previous_in = sign * (previous[axis] - bound) >= 0
        for point in points:
            inside = sign * (point[axis] - bound) >= 0
            if inside != previous_in:
                fraction = (bound - previous[axis]) / (point[axis] - previous[axis])
                clipped.append(previous + fraction * (point - previous))
            if inside:
                clipped.append(point)
            previous, previous_in = point, inside
        points = clipped
    return np.array(points).reshape(-1, 2)


@dataclass(frozen=True)
class ReferenceConfig:
    # Conservative rectangular envelope of the current URDF's cylindrical shoe.
    sole_half_length_m: float = 0.093
    sole_half_width_m: float = 0.036
    edge_margin_m: float = 0.025
    nominal_step_m: float = 0.32
    step_width_m: float = 0.20
    max_step_m: float = 0.55
    max_height_change_m: float = 0.24
    bridge_gap_m: float = 0.30
    surface_tolerance_m: float = 0.04
    anchor_motion_m: float = 0.015
    anchor_rotation_rad: float = 0.20

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in vars(self).values()):
            raise ValueError("Reference settings must be finite and positive")
        if self.nominal_step_m > self.max_step_m:
            raise ValueError("Nominal step must not exceed maximum step")


def arrange_curbs(cases, gaps):
    """Rebuild only curb portions, retaining the original seeded heights/depths.

    Alternating gaps guarantee both decisions in one route. The last curb exits
    onto 1 m of ground. Checkpoints remain on every platform; no goal is relaxed.
    """
    if gaps is None:
        return cases
    if not gaps or not all(math.isfinite(g) and g >= 0.15 for g in gaps):
        raise ValueError("Curb gaps must be finite and at least 0.15 m")
    from terrain_eval_cases import profile_mesh

    result = copy.deepcopy(cases)
    for case in result:
        curbs = case.get("curbs", [])
        if not curbs:
            continue
        start = curbs[0]["start_x_m"]
        profile = [segment for segment in case["profile"] if segment[1] <= start + 1e-8]
        checkpoints = [p for p in case["checkpoints"] if p["terrain"] != "curb"]
        x = start
        for index, curb in enumerate(curbs):
            gap = gaps[index % len(gaps)] if index < len(curbs) - 1 else 1.0
            depth, height = curb["depth_m"], curb["height_m"]
            curb.update(start_x_m=x, end_x_m=x + depth, gap_after_m=gap)
            profile.append([x, x + depth, height, height])
            checkpoints.append({"start_x_m": x, "end_x_m": x + depth,
                                "height_m": height, "terrain": "curb"})
            x += depth
            profile.append([x, x + gap, 0.0, 0.0])
            x += gap
        case.update(stair_end_x_m=x, goal_x_m=x + 0.8, lane_max_x_m=x + 1.6)
        profile.append([x, x + 1.6, 0.0, 0.0])
        case.update(profile=profile, checkpoints=checkpoints)
        case["surface_heights_m"] = [max(lo, hi) for _, _, lo, hi in profile]
        if case["terrain"] == "curb":
            case.update(summit_start_x_m=checkpoints[0]["start_x_m"],
                        summit_end_x_m=checkpoints[0]["end_x_m"],
                        summit_height_m=checkpoints[0]["height_m"])
        case["mesh_vertices"], case["mesh_faces"] = profile_mesh(profile, case["width_m"])
    return result


class LaneGeometry:
    """Exact generated top profile, merging coplanar collision seams."""

    def __init__(self, case, config=None):
        self.case, self.config = case, config or ReferenceConfig()
        if "profile" in case:
            profile = case["profile"]
        else:
            x = case["stair_start_x_m"]
            profile = [[case["lane_min_x_m"], x, case["start_height_m"], case["start_height_m"]]]
            for depth, height in zip(case["tread_depths_m"], case["surface_heights_m"]):
                profile.append([x, x + depth, height, height])
                x += depth
            profile.append([x, case["lane_max_x_m"], case["end_height_m"], case["end_height_m"]])
        self.segments = []
        for left, right, lo, hi in profile:
            slope = (hi - lo) / (right - left)
            if self.segments:
                a, b, z0, z1 = self.segments[-1]
                if abs(left - b) < 1e-8 and abs(lo - z1) < 1e-8 and abs(slope - (z1-z0)/(b-a)) < 1e-8:
                    self.segments[-1] = [a, right, z0, hi]
                    continue
            self.segments.append([left, right, lo, hi])

    def index(self, x):
        for i, (left, right, _, _) in enumerate(self.segments):
            if left <= x < right:
                return i
        return -1

    def height(self, index, x):
        left, right, lo, hi = self.segments[index]
        return lo + (hi-lo) * (x-left)/(right-left)

    def phase(self, x):
        case = self.case
        for i, curb in enumerate(case.get("curbs", [])):
            if curb["start_x_m"] <= x < curb["end_x_m"]:
                return "curb", i
            if curb["end_x_m"] <= x < curb["end_x_m"] + curb["gap_after_m"]:
                return "curb_gap", i
        if case.get("flights") and case["stair_start_x_m"] <= x < case["stair_start_x_m"] + sum(case["tread_depths_m"]):
            if case["direction"] == "down" or (case["direction"] == "up_down" and x >= case["summit_end_x_m"]):
                return "stairs_down", -1
            if case["direction"] == "up_down" and x >= case["summit_start_x_m"]:
                return "stairs_landing", -1
            return "stairs_up", -1
        index = self.index(x)
        if index >= 0 and abs(self.segments[index][3] - self.segments[index][2]) > 1e-8:
            return "slope", -1
        return "ground", -1

    def safe_bounds(self, index):
        left, right, _, _ = self.segments[index]
        inset = self.config.sole_half_length_m + self.config.edge_margin_m
        return left + inset, right - inset

    def curb_rule(self, curb_id):
        """Describe a transition using minimum safe-center stride, not gap alone."""
        c = self.config
        first, second = self.case["curbs"][curb_id:curb_id+2]
        gap = second["start_x_m"] - first["end_x_m"]
        inset = c.sole_half_length_m + c.edge_margin_m
        reach = math.hypot(gap + 2*inset, c.step_width_m)
        height = abs(second["height_m"] - first["height_m"])
        fits = min(first["depth_m"], second["depth_m"]) >= 2*inset
        feasible = fits and reach <= c.max_step_m + 1e-8 and height <= c.max_height_change_m + 1e-8
        mode = "via_ground" if gap > c.bridge_gap_m + 1e-8 else ("bridge_short_gap" if feasible else "bridge_unavailable")
        return {"from_curb": curb_id, "to_curb": curb_id+1, "gap_m": gap,
                "rule": mode, "minimum_horizontal_stride_m": reach, "height_change_m": height,
                "geometric_bridge_feasible": feasible}

    def plan(self, support, foot_id):
        """Freeze an attainable nominal sole-center target before touchdown.

        Reach is a geometric bound, not a dynamic feasibility certificate. Short
        gaps skip the lower ground only when height/stride checks pass. Long gaps
        retain ground as the next support surface.
        """
        c = self.config
        support = np.asarray(support, dtype=float)
        index = self.index(support[0])
        if index < 0:
            return {"valid": False, "reason": "support_outside_lane"}
        y = c.step_width_m / 2 * (1 if foot_id == 0 else -1)
        if abs(y) + c.sole_half_width_m + c.edge_margin_m > self.case["width_m"] / 2:
            return {"valid": False, "reason": "lane_too_narrow"}
        nominal = support[0] + c.nominal_step_m
        _, high = self.safe_bounds(index)
        next_index, route = index + 1, "advance"
        phase, curb_id = self.phase(support[0])
        short_gap = False
        if phase == "curb" and curb_id + 1 < len(self.case.get("curbs", [])):
            curb, following = self.case["curbs"][curb_id:curb_id+2]
            short_gap = curb["gap_after_m"] <= c.bridge_gap_m
            if short_gap:
                next_index = self.index(following["start_x_m"] + 1e-6)
                route = "bridge_short_gap"
            else:
                route = "descend_to_ground"
        elif phase == "curb_gap":
            route = "ground_then_ascend"

        def candidate(segment, x, mode):
            a, b = self.safe_bounds(segment)
            if a > b:
                return None
            x = float(np.clip(x, a, b))
            point = np.array([x, y, self.height(segment, x)])
            if x < support[0] + 0.025 or np.linalg.norm(point[:2] - support[:2]) > c.max_step_m:
                return None
            if abs(point[2] - support[2]) > c.max_height_change_m:
                return None
            return {"valid": True, "reason": "geometric_reference", "target_lane_m": point.tolist(),
                    "surface_id": segment, "route": mode,
                    "target_phase": self.phase(x)[0], "privileged_world_reference": True}

        # Reach the next surface as the current safe interval runs out.
        if nominal >= high and next_index < len(self.segments):
            target = candidate(next_index, max(nominal, self.safe_bounds(next_index)[0]), route)
            if target is not None:
                return target
        target = candidate(index, nominal, "approach_short_gap" if short_gap else "advance")
        if target is not None:
            return target
        return {"valid": False, "reason": "no_reachable_safe_foothold",
                "route": route, "short_gap_requested": short_gap}

    def footprint(self, position, orientation):
        """Signed rectangular-envelope clearance to real top-surface boundaries.

        Negative means projected envelope overhang, not proof of contact at that
        corner. A vertical proximity gate keeps airborne feet out of edge counts.
        """
        c = self.config
        p, r = np.asarray(position), np.asarray(orientation)
        length, width = c.sole_half_length_m, c.sole_half_width_m
        corners = np.array([[-length, -width, 0], [length, -width, 0],
                            [length, width, 0], [-length, width, 0]]) @ r.T + p
        center_index = self.index(p[0])
        index = center_index
        phase, curb_id = self.phase(p[0])
        result = {"surface_id": index, "phase": phase, "curb_id": curb_id,
                  "center_surface_id": center_index, "surface_ambiguous": False,
                  "surface_phase": "unknown", "surface_curb_id": -1,
                  "edge_margin_m": None, "center_height_gap_m": None,
                  "surface_near": False, "edge_status": "outside_lane",
                  "corners_lane_m": corners.tolist()}
        # Select a plausible top underneath ANY part of the sole, not only its
        # center. A heel can remain on the upper tread after the center crosses.
        candidates = []
        normal = r[:, 2]
        min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
        if abs(normal[2]) > 0.25:
            for i, (left, right, _, _) in enumerate(self.segments):
                if right < min_x or left > max_x:
                    continue
                polygon = clip_footprint(corners[:, :2], left, right,
                                         -self.case["width_m"]/2, self.case["width_m"]/2)
                if len(polygon) < 3:
                    continue
                sole_z = p[2] - ((polygon - p[:2]) @ normal[:2])/normal[2]
                gaps = sole_z - np.array([self.height(i, x) for x in polygon[:, 0]])
                proximity = 0.0 if gaps.min() <= 0 <= gaps.max() else float(np.min(abs(gaps)))
                candidates.append((proximity, i))
        if candidates:
            _, index = min(candidates, key=lambda pair: (pair[0], pair[1] != center_index))
        if index < 0:
            return result
        left, right, _, _ = self.segments[index]
        margin = min(corners[:, 0].min()-left, right-corners[:, 0].max(),
                     corners[:, 1].min()+self.case["width_m"]/2,
                     self.case["width_m"]/2-corners[:, 1].max())
        center_gap = p[2] - self.height(index, p[0])
        near_count = sum(distance <= c.surface_tolerance_m for distance, _ in candidates)
        surface_phase, surface_curb = self.phase((left + right)/2)
        result.update(surface_id=index, edge_margin_m=float(margin), center_height_gap_m=float(center_gap),
                      surface_phase=surface_phase, surface_curb_id=surface_curb,
                      surface_near=bool(near_count), surface_ambiguous=near_count > 1,
                      edge_status="overhang" if margin < 0 else ("near_edge" if margin < c.edge_margin_m else "inside"))
        return result
