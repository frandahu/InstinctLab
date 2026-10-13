"""Depth-only local support geometry. No terrain importer or world hit access."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .kinematics import rotate


@dataclass
class VisualFootholdSettings:
    """Version-one parameters; lengths in metres, time in seconds."""

    step_time: float = 0.40
    max_step: float = 0.60
    max_height_change: float = 0.24
    step_width: float = 0.20
    swing_clearance: float = 0.08
    target_sigma: float = 0.08
    resolution: float = 0.04
    memory_s: float = 0.60
    capture_interval: int = 5
    depth_delay_steps: int = 1
    depth_noise_std: float = 0.005
    depth_max: float = 2.5
    joint_position_noise: float = 0.005
    joint_velocity_noise: float = 0.2
    plane_rms_max: float = 0.015
    slope_max: float = 0.45
    replan_interval: int = 5

    def validate(self, dt):
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("Visual foothold settings must be finite and nonnegative")
        if min(self.step_time, self.max_step, self.target_sigma, self.resolution, self.memory_s) <= 0:
            raise ValueError("Step, map and memory parameters must be positive")
        if self.capture_interval < 1 or self.replan_interval < 1 or self.depth_delay_steps != 1:
            raise ValueError("Version one uses a one-control-step depth delay and positive intervals")
        if self.resolution != 0.04 or self.step_time < 4 * dt:
            raise ValueError("Version one footprint stencil uses a 4 cm grid; step time must exceed 4 control ticks")


def depth_points(depth, directions, starts, camera_rotation, camera_position, maximum=2.5):
    """Backproject this project's +X optical-axis planar depth, in metres.

    Inputs are camera-local rays from the actual pattern, never world hit points.
    Invalid/saturated pixels remain unknown. Output is torso-relative XYZ.
    """
    valid = torch.isfinite(depth) & (depth > 0.1) & (depth < maximum - 0.01)
    valid &= directions[..., 0] > 1e-6
    distance = torch.nan_to_num(depth, nan=0., posinf=0., neginf=0.) / directions[..., 0].clamp_min(1e-6)
    points = starts + directions * distance[..., None]
    points = rotate(camera_rotation, points) + camera_position
    return points, valid


class SupportGrid:
    """Small local map with explicit unknowns and conservative footprint tests.

    At most one-cell holes with >=4 observed neighbours and consistent heights
    are interpolated. This is a depth estimate, not a support safety certificate.
    The 28x20 cm axis-aligned stencil includes the nominal shoe and margin for
    heading changes up to 20 degrees relative to the current anchor axes.
    """

    def __init__(self, settings, device):
        self.cfg = settings
        self.x0, self.y0, self.nx, self.ny = -0.4, -0.68, 41, 35
        y, x = torch.meshgrid(
            torch.arange(self.ny, device=device), torch.arange(self.nx, device=device), indexing="ij"
        )
        self.xy = torch.stack((x.flatten() * settings.resolution + self.x0,
                               y.flatten() * settings.resolution + self.y0), -1)
        yy, xx = torch.meshgrid(torch.arange(-2, 3, device=device), torch.arange(-3, 4, device=device), indexing="ij")
        self.kernel = torch.ones(1, 1, 5, 7, device=device) / 35
        self.kx = (xx * settings.resolution)[None, None] / (xx.square().sum() * settings.resolution**2)
        self.ky = (yy * settings.resolution)[None, None] / (yy.square().sum() * settings.resolution**2)
        self.var_x = (xx.float() * settings.resolution).square().mean()
        self.var_y = (yy.float() * settings.resolution).square().mean()

    def build(self, points, valid):
        """Rasterize samples and return height, observed mask and footprint mask."""
        n, _, _ = points.shape
        cells = self.nx * self.ny
        ix = ((points[..., 0] - self.x0) / self.cfg.resolution).round().long()
        iy = ((points[..., 1] - self.y0) / self.cfg.resolution).round().long()
        valid = valid & torch.isfinite(points).all(-1) & (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        index = (iy.clamp(0, self.ny-1) * self.nx + ix.clamp(0, self.nx-1))
        weight = valid.float()
        z = torch.where(valid, points[..., 2], 0.)
        count = torch.zeros(n, cells, device=points.device).scatter_add_(1, index, weight)
        total = torch.zeros_like(count).scatter_add_(1, index, z * weight)
        square = torch.zeros_like(count).scatter_add_(1, index, z.square() * weight)
        obstacle_h = torch.full_like(count, -float("inf")).scatter_reduce_(
            1, index, z.masked_fill(~valid, -float("inf")), reduce="amax", include_self=True
        )
        obstacle_seen = count > 0
        obstacle_h = torch.where(obstacle_seen, obstacle_h, 0.)
        h = (total / count.clamp_min(1)).view(n, 1, self.ny, self.nx)
        variance = (square / count.clamp_min(1) - (total / count.clamp_min(1)).square()).clamp_min(0)
        observed = ((count > 0) & (variance < self.cfg.plane_rms_max**2)).view_as(h)
        neighbour = torch.ones(1, 1, 3, 3, device=points.device)
        hits = F.conv2d(observed.float(), neighbour, padding=1)
        mean = F.conv2d(h * observed, neighbour, padding=1) / hits.clamp_min(1)
        spread = F.conv2d(h.square() * observed, neighbour, padding=1) / hits.clamp_min(1) - mean.square()
        fill = ~observed & (hits >= 4) & (spread < self.cfg.plane_rms_max**2)
        h = torch.where(fill, mean, h)
        observed |= fill
        coverage = F.conv2d(observed.float(), self.kernel, padding=(2, 3))
        average = F.conv2d(h, self.kernel, padding=(2, 3))
        bx = F.conv2d(h, self.kx, padding=(2, 3))
        by = F.conv2d(h, self.ky, padding=(2, 3))
        residual = (F.conv2d(h.square(), self.kernel, padding=(2, 3)) - average.square()
                    - bx.square() * self.var_x - by.square() * self.var_y).clamp_min(0)
        safe = (coverage > 0.999) & (residual < self.cfg.plane_rms_max**2)
        safe &= (bx.square() + by.square() < math.tan(self.cfg.slope_max)**2)
        return h.flatten(1), observed.flatten(1), safe.flatten(1), obstacle_h, obstacle_seen

    def sample(self, values, xy):
        """Sample grid values at arbitrary XY; outside the map is zero/unknown."""
        grid = xy.clone()
        grid[..., 0] = 2 * (grid[..., 0] - self.x0) / ((self.nx-1) * self.cfg.resolution) - 1
        grid[..., 1] = 2 * (grid[..., 1] - self.y0) / ((self.ny-1) * self.cfg.resolution) - 1
        return F.grid_sample(values.float().view(-1, 1, self.ny, self.nx), grid[:, None],
                             mode="nearest", align_corners=True).squeeze(1).squeeze(1)

    def choose(self, points, valid, support, swing, direction, speed, swing_start, offset_xy=None):
        """Minimize cost over patches satisfying *all* geometry/coverage tests.

        Support defines reachability; the actual swing foot defines the corridor,
        matching the controller's reference. Unknown middle/end samples reject a
        candidate even when its landing footprint is observed. Input samples may
        include unexpired, reliably transformed history, never invented ground.
        """
        h, observed, safe, obstacle_h, obstacle_seen = self.build(points, valid)
        xy = self.xy[None].expand(len(points), -1, -1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        lateral = torch.stack((-direction[:, 1], direction[:, 0]), -1)
        side = torch.where(swing == 0, 1., -1.)
        nominal = support[:, :2] + direction * (speed * self.cfg.step_time).clamp(0.24, 0.40)[:, None]
        nominal += lateral * side[:, None] * self.cfg.step_width
        delta = xy - support[:, None, :2]
        forward = (delta * direction[:, None]).sum(-1)
        sideways = (delta * lateral[:, None]).sum(-1) * side[:, None]
        feasible = safe & (delta.norm(dim=-1) <= self.cfg.max_step)
        feasible &= (forward >= 0.08) & (sideways >= 0.10) & (sideways <= 0.32)
        feasible &= (h - support[:, 2, None]).abs() <= self.cfg.max_height_change
        # The rectangular support stencil is conservative only in this yaw range.
        feasible &= (direction[:, 0] > math.cos(math.radians(20)))[:, None]
        reachable_count = feasible.sum(-1)
        # Bound temporary storage for large training batches. Test each candidate
        # before ranking, rather than rejecting only the cheapest landing patch.
        corridor_ok = torch.zeros_like(feasible)
        required_phase = torch.linspace(0, 1, 9, device=points.device)[3:][None, None, :, None]
        observed_samples = torch.zeros(len(points), device=points.device)
        for begin in range(0, self.nx * self.ny, 128):
            end = min(begin + 128, self.nx * self.ny)
            path = swing_start[:, None, None, :2] * (1-required_phase)
            path = path + xy[:, begin:end, None] * required_phase
            seen = self.sample(obstacle_seen, path.flatten(1, 2)).reshape(len(points), end-begin, 6) > 0.5
            corridor_ok[:, begin:end] = seen.all(-1)
            observed_samples += (seen & feasible[:, begin:end, None]).sum((1, 2))
        feasible &= corridor_ok
        self.diagnostics = {
            "safe_candidates": safe.sum(-1),
            "reachable_candidates": reachable_count,
            "corridor_candidates": feasible.sum(-1),
            "no_observations": ~valid.any(-1),
            "no_safe_candidates": valid.any(-1) & ~safe.any(-1),
            "feasibility_failures": safe.any(-1) & (reachable_count == 0),
            "corridor_unknown": (reachable_count > 0) & ~feasible.any(-1),
            "corridor_required_samples": reachable_count * 6,
            "corridor_observed_samples": observed_samples,
        }
        cost = (xy - nominal[:, None]).square().sum(-1)
        # Same-height platforms win over dropping into short curb gaps when reachable.
        cost += 0.6 * (h - support[:, 2, None]).abs()
        cost = cost.masked_fill(~feasible, float("inf"))
        best = cost.argmin(-1)
        rows = torch.arange(len(points), device=points.device)
        planned = torch.cat((xy[rows, best], h[rows, best, None]), -1)
        target_valid = feasible.any(-1)
        requested = planned.clone()
        if offset_xy is not None:
            if offset_xy.shape != (len(points), 2) or not torch.isfinite(offset_xy).all():
                raise ValueError("Reference offset must be finite anchor-frame XY [batch,2]")
            requested[:, :2] += offset_xy
            # Project onto the SAME fully feasible candidates, including the
            # original footprint, reach, yaw, height and observed corridor rules.
            # Prefer the planned height; this is a finite-grid projection, not a
            # continuous safety certificate. Ties keep the unmodified plan.
            distance = (xy - requested[:, None, :2]).square().sum(-1)
            distance += 0.6 * (h - planned[:, 2, None]).abs()
            distance.masked_fill_(~feasible, float("inf"))
            projected = distance.argmin(-1)
            keep = distance[rows, best] <= distance[rows, projected] + 1e-8
            projected = torch.where(keep, best, projected)
            # Baseline must retain the original ranking, even on mixed heights.
            best = torch.where(offset_xy.norm(dim=-1) > 1e-8, projected, best)
        target = torch.cat((xy[rows, best], h[rows, best, None]), -1)
        self.reference = {"planned": planned, "requested": requested, "applied": target, "valid": target_valid}
        # Check observed swing corridor; unseen space is not silently treated as clear.
        phase = torch.linspace(0, 1, 9, device=points.device)[None, :, None]
        path_xy = swing_start[:, None, :2] * (1-phase) + target[:, None, :2] * phase
        # A depth-discontinuity cell is unsuitable for support, but its highest
        # observed point still contributes to swing clearance. Do not erase the
        # stair riser from the obstacle map when rejecting its mixed-height cell.
        path_h = self.sample(obstacle_h, path_xy)
        path_seen = self.sample(obstacle_seen, path_xy) > 0.5
        # Coverage has already filtered every candidate. Initial swing vicinity
        # may be out of view; the existing middle/end coverage rule is unchanged.
        apex = torch.maximum(path_h.masked_fill(~path_seen, -float("inf")).amax(-1),
                             torch.maximum(target[:, 2], swing_start[:, 2]))
        return target, target_valid, apex + self.cfg.swing_clearance
