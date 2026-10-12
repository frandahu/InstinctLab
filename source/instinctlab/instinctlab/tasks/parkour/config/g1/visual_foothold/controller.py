"""Batched foothold state, short contact-anchored memory, and locked commands.

All spatial inputs are torso-relative or contact measurements. No world robot
position, terrain mesh, terrain label, or case geometry is accepted here.
"""

import math

import torch
import torch.nn.functional as F

from .kinematics import ContactObserver, rotate
from .perception import SupportGrid


class VisualFootholdController:
    """Shared state advanced once per control step by the environment adapter."""

    def __init__(self, n, pixels, dt, settings, device):
        settings.validate(dt)
        self.cfg, self.dt, self.n = settings, dt, n
        self.contact = ContactObserver(n, dt, device)
        self.grid = SupportGrid(settings, device)
        self.rows = torch.arange(n, device=device)
        self.eye = torch.eye(3, device=device).expand(n, 3, 3)
        slots = math.ceil(settings.memory_s / (settings.capture_interval * dt)) + 1
        self.points = torch.zeros(n, slots, pixels, 3, device=device)
        self.point_valid = torch.zeros(n, slots, pixels, dtype=torch.bool, device=device)
        self.point_time = torch.full((n, slots), -float("inf"), device=device)
        self.pending = torch.zeros(n, pixels, 3, device=device)
        self.pending_valid = torch.zeros(n, pixels, dtype=torch.bool, device=device)
        self.pending_time = torch.zeros(n, device=device)
        self.time = torch.zeros(n, device=device)
        self.anchor = torch.full((n,), -1, dtype=torch.long, device=device)
        self.anchor_offset = self.eye.clone()
        self.target = torch.zeros(n, 3, device=device)
        self.start = torch.zeros_like(self.target)
        self.start_time = torch.zeros(n, device=device)
        self.apex = torch.zeros(n, device=device)
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.landed = torch.zeros_like(self.valid)
        self.slot = torch.full((n,), -1, dtype=torch.long, device=device)
        self.swing = torch.zeros(n, dtype=torch.long, device=device)
        self.candidate_error = torch.zeros(n, 2, device=device)
        self.candidate_valid = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.candidate_slot = torch.full((n, 2), -1, dtype=torch.long, device=device)
        self.candidate_risk = torch.zeros(n, 2, device=device)
        self.candidate_known = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.paid_slot = torch.full((n, 2), -1, dtype=torch.long, device=device)
        self.previous_relative = torch.zeros(n, 3, device=device)
        self.observation = torch.zeros(n, 41, device=device)
        self.reward = {name: torch.zeros(n, device=device) for name in ("touchdown", "swing", "edge", "miss")}
        self.stats = {name: torch.zeros(n, device=device) for name in (
            "ticks", "valid_ticks", "support_ticks", "touchdowns", "commanded_touchdowns", "error_sum_m",
            "hits", "visual_risk", "missed_steps", "anchor_losses",
            "moving_ticks", "target_exposure_ticks", "target_retained_ticks", "planning_attempts", "targets_generated",
            "safe_candidates", "reachable_candidates", "corridor_candidates", "no_observations", "no_safe_candidates",
            "feasibility_failures", "corridor_unknown", "corridor_required_samples", "corridor_observed_samples",
            "memory_frames_used", "history_points_used", "targets_cleared", "clear_anchor_loss", "clear_transfer",
            "clear_phase", "clear_stopped", "cleared_unlanded", "touchdown_candidates", "candidate_aborts",
            "confirmed_commanded_touchdowns", "reward_payments", "duplicate_payments_blocked", "risk_samples",
            "reward_touchdown_integral", "reward_swing_integral", "reward_edge_integral", "reward_miss_integral",
        )}
        self.last_error = torch.zeros(n, device=device)
        self.last_touchdown = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.contact.reset(ids)
        self.anchor[ids] = -1
        self.anchor_offset[ids] = self.eye[ids]
        self.point_valid[ids] = False
        self.pending_valid[ids] = False
        self.point_time[ids] = -float("inf")
        self.time[ids] = 0
        self.valid[ids] = False
        self.landed[ids] = False
        self.slot[ids] = -1
        self.candidate_valid[ids] = False
        self.candidate_slot[ids] = -1
        self.paid_slot[ids] = -1
        self.candidate_known[ids] = False
        self.target[ids] = 0
        self.start[ids] = 0
        self.observation[ids] = 0
        self.last_touchdown[ids] = False
        self.last_error[ids] = 0
        for tensor in (*self.stats.values(), *self.reward.values()):
            tensor[ids] = 0

    @torch.no_grad()
    def update(self, p, rotation, velocity, force, gravity_b, command, points_b, point_valid, advance, tick):
        """Consume one observation; advance=False bootstraps reset rows without time travel."""
        fresh = self.contact.fresh.clone()
        active = advance | fresh
        previous_valid = self.valid.clone()
        candidate, touchdown = self.contact.update(velocity, force, advance)
        support = self.contact.state == 2
        self.time += advance.float() * self.dt
        scheduled_slot = torch.floor((self.time + 1e-6) / self.cfg.step_time).long()
        scheduled_change = active & (scheduled_slot != self.slot)
        phase = ((self.time - scheduled_slot.float() * self.cfg.step_time) / self.cfg.step_time).clamp(0, 1)
        for value in self.reward.values():
            value[active] = 0
        self.last_touchdown[active] = False
        old = self.anchor.clone()
        old_index = old.clamp_min(0)
        old_ok = (old >= 0) & support[self.rows, old_index]
        selected = torch.where(old_ok, old, support.long().argmax(-1))
        # Anchor the opposite foot BEFORE scheduled liftoff, while both feet are
        # available. Anchoring the swing foot would discard the locked command
        # as soon as that foot leaves the ground.
        preferred = 1 - scheduled_slot % 2
        choose_stance = (scheduled_change | fresh) & support[self.rows, preferred]
        selected = torch.where(choose_stance, preferred, selected)
        # Transfer during double support, when the new foot is confirmed, not after release.
        new_contact = touchdown.any(-1) & old_ok
        selected = torch.where(new_contact, touchdown.long().argmax(-1), selected)
        selected = torch.where(support.any(-1), selected, -1)
        selected = torch.where(active, selected, old)
        new_index = selected.clamp_min(0)
        old_r = rotation[self.rows, old_index] @ self.anchor_offset
        old_p = p[self.rows, old_index]
        z = -gravity_b / gravity_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = torch.zeros_like(z)
        x[:, 0] = 1
        x = x - (x * z).sum(-1, keepdim=True) * z
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        heading = torch.stack((x, torch.cross(z, x, dim=-1), z), -1)
        changed = active & (selected != old)
        relative = rotate(rotation[:, 0].transpose(-1, -2), p[:, 1] - p[:, 0])
        double_consistent = (relative - self.previous_relative).norm(dim=-1) < 0.015
        transferable = changed & old_ok & support[self.rows, new_index] & double_consistent & ~fresh
        self.anchor_offset[changed] = (rotation[self.rows, new_index].transpose(-1, -2) @ heading)[changed]
        r = rotation[self.rows, new_index] @ self.anchor_offset
        origin = p[self.rows, new_index]
        transform_r = r.transpose(-1, -2) @ old_r
        transform_p = rotate(r.transpose(-1, -2), old_p - origin)
        # A confirmed stance is only a no-slip proxy. Reject substantial foot roll.
        upright_anchor = (r[..., 2] * z).sum(-1) > math.cos(0.20)
        anchor_ok = (selected >= 0) & upright_anchor
        lost = active & ((old >= 0) & ~old_ok & ~transferable | ~anchor_ok)
        clear = active & ((changed & ~transferable) | lost | fresh)
        cleared = clear & previous_valid
        self.stats["clear_anchor_loss"] += (cleared & lost).float()
        self.stats["clear_transfer"] += (cleared & ~lost).float()
        self.stats["cleared_unlanded"] += (cleared & ~self.landed).float()
        self.point_valid[clear] = False
        self.point_time[clear] = -float("inf")
        self.pending_valid[clear] = False
        self.valid[clear] = False
        # Candidate error/risk are scalar measurements latched in the old frame.
        # Their lifecycle ends on confirmation, contact abort or episode reset,
        # independently of map/target invalidation during an anchor transfer.
        # Re-express old data only at a valid support transfer. Otherwise discard.
        self.points[transferable] = (rotate(transform_r[:, None, None], self.points)
                                    + transform_p[:, None, None])[transferable]
        self.pending[transferable] = (rotate(transform_r[:, None], self.pending) + transform_p[:, None])[transferable]
        self.target[transferable] = (rotate(transform_r, self.target) + transform_p)[transferable]
        self.start[transferable] = (rotate(transform_r, self.start) + transform_p)[transferable]
        # Transform an upper height bound over the local grid, including small
        # anchor tilts; a scalar translation alone would underestimate clearance.
        transformed_apex = (transform_r[:, 2, 2] * self.apex + transform_p[:, 2]
                            + transform_r[:, 2, 0].abs()*1.2 + transform_r[:, 2, 1].abs()*0.68)
        self.apex[transferable] = transformed_apex[transferable]
        self.anchor = torch.where(anchor_ok, selected, -1)
        self.previous_relative[active] = relative[active]
        self.stats["anchor_losses"] += (lost & (old >= 0)).float()
        p_a = rotate(r[:, None].transpose(-1, -2), p - origin[:, None])
        cloud = rotate(r[:, None].transpose(-1, -2), points_b - origin[:, None])
        if tick % self.cfg.capture_interval == 0:
            index = (tick // self.cfg.capture_interval) % self.points.shape[1]
            self.points[active, index] = self.pending[active]
            self.point_valid[active, index] = (self.pending_valid & anchor_ok[:, None])[active]
            self.point_time[active, index] = self.pending_time[active]
        self.pending[active] = cloud[active]
        self.pending_valid[active] = (point_valid & anchor_ok[:, None])[active]
        self.pending_time[active] = self.time[active]
        memory_valid = self.point_valid & ((self.time[:, None] - self.point_time) <= self.cfg.memory_s)[..., None]
        flat_points = self.points.flatten(1, 2)
        flat_valid = memory_valid.flatten(1, 2)
        next_slot = scheduled_slot
        phase_change = active & (next_slot != self.slot)
        moving = command[:, :2].norm(dim=-1) > 0.15
        missed = phase_change & self.valid & ~self.landed & moving & ~self.candidate_valid.any(-1)
        self.reward["miss"][missed] = 1 / self.dt
        self.stats["missed_steps"] += missed.float()
        phase_clear = phase_change & self.valid
        self.stats["clear_phase"] += phase_clear.float()
        cleared |= phase_clear
        self.valid[phase_change] = False
        self.landed[phase_change] = False
        self.slot[active] = next_slot[active]
        self.swing[active] = (next_slot % 2)[active]
        other = 1 - self.swing
        planning = active & moving & anchor_ok & support[self.rows, other] & ~self.valid
        planning &= phase_change | (tick % self.cfg.replan_interval == 0)
        planning &= phase < 0.31
        ids = planning.nonzero().flatten()
        if len(ids):
            direction_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), -1)
            direction_a = rotate(r.transpose(-1, -2), direction_b)[:, :2]
            target, valid, apex = self.grid.choose(
                flat_points[ids], flat_valid[ids], p_a[ids, other[ids]], self.swing[ids],
                direction_a[ids], command[ids, :2].norm(dim=-1),
                p_a[ids, self.swing[ids]],
            )
            self.stats["planning_attempts"][ids] += 1
            self.stats["targets_generated"][ids] += valid.float()
            for name, value in self.grid.diagnostics.items():
                self.stats[name][ids] += value.float()
            self.stats["memory_frames_used"][ids] += memory_valid[ids].any(-1).sum(-1)
            historical = (self.time[:, None] - self.point_time) > self.cfg.capture_interval * self.dt + 1e-6
            self.stats["history_points_used"][ids] += (memory_valid[ids] & historical[ids, :, None]).sum((1, 2))
            self.target[ids] = target
            self.start[ids] = p_a[ids, self.swing[ids]]
            self.start_time[ids] = self.time[ids]
            self.apex[ids] = apex
            self.valid[ids] = valid
        stop_clear = active & self.valid & ~moving
        self.stats["clear_stopped"] += stop_clear.float()
        cleared |= stop_clear
        self.stats["targets_cleared"] += cleared.float()
        self.valid = torch.where(active, self.valid & anchor_ok & moving, self.valid)
        # Candidate pose/error is latched now, confirmed 20 ms later, and paid once.
        wanted = F.one_hot(self.swing, 2).bool()
        latch = candidate & wanted & self.valid[:, None] & ~self.landed[:, None]
        latch &= self.paid_slot != self.slot[:, None]
        error = (p_a - self.target[:, None]).norm(dim=-1)
        self.candidate_valid[candidate] = latch[candidate]
        self.candidate_slot[latch] = self.slot[:, None].expand(-1, 2)[latch]
        self.candidate_error[latch] = error[latch]
        event_ids = latch.any(-1).nonzero().flatten()
        if len(event_ids):
            _, seen, safe, _, _ = self.grid.build(flat_points[event_ids], flat_valid[event_ids])
            risk = 1 - self.grid.sample(safe, p_a[event_ids, ..., :2])
            known = self.grid.sample(seen, p_a[event_ids, ..., :2]) > 0.5
            self.candidate_risk[event_ids] = torch.where(latch[event_ids], risk * known, self.candidate_risk[event_ids])
            self.candidate_known[event_ids] = torch.where(latch[event_ids], known, self.candidate_known[event_ids])
        confirmed = touchdown & self.candidate_valid
        paid = confirmed & (self.paid_slot != self.candidate_slot)
        score = torch.exp(-self.candidate_error.square() / self.cfg.target_sigma**2)
        self.reward["touchdown"][active] = (score * paid).sum(-1)[active] / self.dt
        self.reward["edge"][active] = (self.candidate_risk * paid).sum(-1)[active] / self.dt
        paid_any = paid.any(-1)
        self.landed |= (paid & (self.candidate_slot == self.slot[:, None])).any(-1)
        self.last_touchdown[active] = paid_any[active]
        self.last_error[paid_any] = (self.candidate_error * paid).sum(-1)[paid_any]
        self.stats["touchdowns"] += touchdown.sum(-1)
        self.stats["touchdown_candidates"] += latch.sum(-1)
        self.stats["confirmed_commanded_touchdowns"] += confirmed.sum(-1)
        self.stats["reward_payments"] += paid.sum(-1)
        self.stats["duplicate_payments_blocked"] += (confirmed & ~paid).sum(-1)
        self.stats["commanded_touchdowns"] += paid.sum(-1)
        self.stats["error_sum_m"] += (self.candidate_error * paid).sum(-1)
        self.stats["hits"] += ((self.candidate_error < 0.05) & paid).sum(-1)
        self.stats["visual_risk"] += (self.candidate_risk * paid).sum(-1)
        self.stats["risk_samples"] += (self.candidate_known & paid).sum(-1)
        self.stats["candidate_aborts"] += (self.candidate_valid & (self.contact.state == 0)).sum(-1)
        self.paid_slot[paid] = self.candidate_slot[paid]
        self.candidate_valid[touchdown | (self.contact.state == 0)] = False
        u = phase
        end_time = (self.slot + 1).float() * self.cfg.step_time
        progress = ((self.time - self.start_time) / (end_time - self.start_time).clamp_min(self.dt)).clamp(0, 1)
        reference = self.start * (1-progress[:, None]) + self.target * progress[:, None]
        lift = (self.apex - torch.minimum(self.start[:, 2], self.target[:, 2])).clamp(0.04, 0.40)
        reference[:, 2] += lift * torch.sin(math.pi * progress)
        swing_error = (p_a[self.rows, self.swing] - reference).square().sum(-1)
        swing_mask = self.valid & ~self.landed & (u > 0.1) & (u < 0.9)
        self.reward["swing"][active] = (torch.exp(-swing_error / 0.10**2) * swing_mask)[active]
        target_b = rotate(r, self.target) + origin
        target_error_b = target_b - p[self.rows, self.swing]
        # 41 scalars: p(6), v(6), FSM one-hot(8), support(2), target(3), error(3),
        # swing foot(2), phase sin/cos(2), valid(1), anchor valid(1), ref(3), gravity(3), age(1).
        ref_b = rotate(r, reference) + origin
        age = (self.time - self.point_time.max(-1).values).clamp(0, self.cfg.memory_s) / self.cfg.memory_s
        observation = torch.cat((
            p.flatten(1), self.contact.velocity.flatten(1) * 0.2,
            F.one_hot(self.contact.state, 4).float().flatten(1), support.float(),
            target_b * self.valid[:, None], target_error_b * self.valid[:, None],
            wanted.float(), torch.stack((torch.sin(2*math.pi*u), torch.cos(2*math.pi*u)), -1),
            self.valid[:, None].float(), anchor_ok[:, None].float(), ref_b * self.valid[:, None], gravity_b,
            age[:, None],
        ), -1)
        self.observation[active] = torch.nan_to_num(observation[active]).clamp(-10, 10)
        self.stats["ticks"] += advance.float()
        self.stats["valid_ticks"] += (advance & self.valid).float()
        self.stats["support_ticks"] += (advance & anchor_ok).float()
        self.stats["moving_ticks"] += (advance & moving).float()
        self.stats["target_exposure_ticks"] += (advance & previous_valid).float()
        self.stats["target_retained_ticks"] += (advance & previous_valid & self.valid & ~cleared).float()
        for name, value in self.reward.items():
            self.stats["reward_" + name + "_integral"] += value * advance.float() * self.dt

    def episode_metrics(self, ids):
        """Log additive totals; derive event ratios from totals, never zero-fill errors.

        Keys stay fixed for runner aggregation. TensorBoard raw totals can be
        averaged by the runner: divide matching aggregated numerator/denominator
        tags, rather than averaging per-episode ratios. Playback emits JSON null
        for absent samples. Availability tags explicitly mark empty reset groups.
        """
        s = {name: value[ids].sum() for name, value in self.stats.items()}
        metrics = {"raw/" + name: value for name, value in s.items()}
        metrics.update(
            episodes=s["ticks"].new_tensor(len(ids)),
            ticks_available=(s["ticks"] > 0).float(),
            touchdown_samples_available=(s["commanded_touchdowns"] > 0).float(),
            risk_samples_available=(s["risk_samples"] > 0).float(),
            valid_target_fraction=s["valid_ticks"] / s["ticks"].clamp_min(1),
            support_fraction=s["support_ticks"] / s["ticks"].clamp_min(1),
        )
        return metrics
