"""Causal reference experiments and signed, first-contact execution measurements.

Positions in an event use the contact anchor frame at FIRST contact. EWMA uses
gravity-levelled torso heading XY, so an anchor transfer cannot rotate or
translate old errors into a different meaning. Neither path uses world pose.
"""

import math
from dataclasses import dataclass

import torch

from .kinematics import rotate


@dataclass
class ReferenceExperimentSettings:
    """Online training feedback and playback; lengths in metres, alpha per touchdown."""

    mode: str = "compensate"
    perturb_axis: str = "x"
    perturb_m: float = 0.02
    ewma_alpha: float = 0.2
    min_samples: int = 1
    max_compensation_m: float = 0.04
    max_update_error_m: float = 0.50
    erased_tolerance_m: float = 0.001

    def validate(self):
        if self.mode not in ("baseline", "perturb", "compensate") or self.perturb_axis not in ("x", "y"):
            raise ValueError("Unknown reference experiment mode or perturbation axis")
        if not all(math.isfinite(v) for v in (
            self.perturb_m, self.ewma_alpha, self.max_compensation_m,
            self.max_update_error_m, self.erased_tolerance_m,
        )):
            raise ValueError("Reference experiment parameters must be finite")
        if self.mode == "perturb" and not any(abs(self.perturb_m - v) < 1e-8 for v in (-0.04, -0.02, 0.02, 0.04)):
            raise ValueError("Perturbation experiments support +/-2 cm and +/-4 cm")
        if not 0 < self.ewma_alpha <= 1 or type(self.min_samples) is not int or self.min_samples < 1:
            raise ValueError("EWMA requires alpha in (0,1] and positive integer min_samples")
        if not 0 <= self.max_compensation_m <= 0.04 or self.max_update_error_m <= 0:
            raise ValueError("Compensation must be within 4 cm; update error limit must be positive")
        if not 0 < self.erased_tolerance_m < 0.02:
            raise ValueError("Erased-correction tolerance must be between 0 and 2 cm")


class ReferenceExperiment:
    """Per-env/per-foot estimator and immutable candidate-contact event records.

    Call begin_tick once at the controller update boundary, finish previously
    latched contacts, then propose/commit and latch new contacts. Observation/
    reward readers only consume cached tensors. A newly confirmed error can
    first affect a subsequent planning call, including at a phase boundary.
    """

    def __init__(self, n, device, settings=None):
        self.cfg = settings or ReferenceExperimentSettings()
        self.cfg.validate()
        self.bias = torch.zeros(n, 2, 2, device=device)
        self.samples = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.version = torch.zeros(n, dtype=torch.long, device=device)
        self.plan_event = torch.zeros(n, dtype=torch.bool, device=device)
        self.event_status = torch.zeros(n, 2, dtype=torch.long, device=device)  # 1 confirmed, 2 aborted
        # All plan positions are transported together with the live command.
        self.plan = {name: torch.zeros(n, 3, device=device) for name in ("planned", "requested", "applied")}
        self.plan.update({
            "reference_id": self.version.clone(),
            "slot": self.version.clone(),
            "foot": self.version.clone(),
            "anchor": self.version.clone(),
            "plan_time_s": torch.zeros(n, device=device),
            "bias_heading": torch.zeros(n, 2, device=device),
            "bias_samples": self.version.clone(),
            "correction_requested": self.plan_event.clone(),
            "projection_changed": self.plan_event.clone(),
            "correction_erased": self.plan_event.clone(),
            "correction_opposed": self.plan_event.clone(),
        })
        self.candidate = {
            name: torch.zeros((n, 2) + tuple(value.shape[1:]), dtype=value.dtype, device=device)
            for name, value in self.plan.items()
        }
        self.candidate.update({
            "actual": torch.zeros(n, 2, 3, device=device),
            "error": torch.zeros(n, 2, 3, device=device),
            "error_heading": torch.zeros(n, 2, 2, device=device),
            "contact_time_s": torch.zeros(n, 2, device=device),
            "confirm_time_s": torch.zeros(n, 2, device=device),
            "estimator_accepted": self.event_status.bool(),
        })
        self.totals = {name: torch.zeros(n, device=device) for name in (
            "reference_commands", "corrections_requested", "projection_changed", "corrections_erased",
            "corrections_opposed", "requested_delta_sum_m", "applied_delta_sum_m", "projection_delta_sum_m",
            "signed_error_x_sum_m", "signed_error_y_sum_m", "abs_error_x_sum_m", "abs_error_y_sum_m",
            "error_xy_sum_m", "signed_error_samples", "ewma_accepted", "ewma_rejected",
        )}

    def reset(self, ids):
        self.bias[ids] = 0
        self.samples[ids] = 0
        self.version[ids] = 0
        self.plan_event[ids] = False
        self.event_status[ids] = 0
        for value in (*self.plan.values(), *self.candidate.values(), *self.totals.values()):
            value[ids] = 0

    def begin_tick(self, active):
        self.plan_event[active] = False
        self.event_status[active] = 0

    def propose(self, ids, feet, anchor_from_heading):
        """Return anchor XY offset using strictly past, confirmed measurements."""
        offset = torch.zeros(len(ids), 2, device=self.bias.device)
        if self.cfg.mode == "perturb":
            # Perturbations are in the live anchor axes, matching the grid.
            offset[:, 0 if self.cfg.perturb_axis == "x" else 1] = self.cfg.perturb_m
        elif self.cfg.mode == "compensate":
            bias = self.bias[ids, feet]
            scale = (self.cfg.max_compensation_m / bias.norm(dim=-1).clamp_min(1e-8)).clamp_max(1)
            correction = -bias * scale[:, None]
            correction *= (self.samples[ids, feet] >= self.cfg.min_samples)[:, None]
            offset = rotate(anchor_from_heading, torch.cat((correction, torch.zeros_like(correction[:, :1])), -1))[:, :2]
        return offset

    def commit(self, ids, feet, geometry, time, slot, anchor):
        """Record the unmodified plan, proposal and safely projected command."""
        valid = geometry["valid"]
        ids, feet = ids[valid], feet[valid]
        if not len(ids):
            return
        self.version[ids] += 1
        self.plan_event[ids] = True
        for name in ("planned", "requested", "applied"):
            self.plan[name][ids] = geometry[name][valid]
        self.plan["reference_id"][ids] = self.version[ids]
        self.plan["foot"][ids] = feet
        self.plan["slot"][ids] = slot[ids]
        self.plan["anchor"][ids] = anchor[ids]
        self.plan["plan_time_s"][ids] = time[ids]
        self.plan["bias_heading"][ids] = self.bias[ids, feet]
        self.plan["bias_samples"][ids] = self.samples[ids, feet]
        requested = self.plan["requested"][ids, :2] - self.plan["planned"][ids, :2]
        applied = self.plan["applied"][ids, :2] - self.plan["planned"][ids, :2]
        projection = (self.plan["applied"][ids] - self.plan["requested"][ids]).norm(dim=-1)
        correction = requested.norm(dim=-1) > self.cfg.erased_tolerance_m
        erased = correction & (applied.norm(dim=-1) <= self.cfg.erased_tolerance_m)
        opposed = correction & ~erased & ((requested * applied).sum(-1) <= 0)
        changed = projection > self.cfg.erased_tolerance_m
        for name, value in (("correction_requested", correction), ("projection_changed", changed),
                            ("correction_erased", erased), ("correction_opposed", opposed)):
            self.plan[name][ids] = value
        for name, value in (
            ("reference_commands", torch.ones_like(projection)), ("corrections_requested", correction),
            ("projection_changed", changed), ("corrections_erased", erased), ("corrections_opposed", opposed),
            ("requested_delta_sum_m", requested.norm(dim=-1)), ("applied_delta_sum_m", applied.norm(dim=-1)),
            ("projection_delta_sum_m", projection),
        ):
            self.totals[name][ids] += value.float()

    def transform(self, mask, rotation, translation):
        for name in ("planned", "requested", "applied"):
            self.plan[name][mask] = (rotate(rotation, self.plan[name]) + translation)[mask]
        # Candidate measurements intentionally stay in their first-contact frame.

    def latch(self, mask, actual, applied, heading_from_anchor, time, anchor):
        for name, value in self.plan.items():
            self.candidate[name][mask] = value[:, None].expand_as(self.candidate[name])[mask]
        error = actual - applied[:, None]
        self.candidate["applied"][mask] = applied[:, None].expand_as(actual)[mask]
        self.candidate["actual"][mask] = actual[mask]
        self.candidate["error"][mask] = error[mask]
        self.candidate["error_heading"][mask] = rotate(heading_from_anchor[:, None], error)[..., :2][mask]
        self.candidate["contact_time_s"][mask] = time[:, None].expand_as(mask)[mask]
        self.candidate["confirm_time_s"][mask] = 0
        self.candidate["anchor"][mask] = anchor[:, None].expand_as(mask)[mask]
        self.candidate["estimator_accepted"][mask] = False

    def finish(self, confirmed, aborted, time):
        """Publish once, then update each foot's EWMA; aborts never train it."""
        self.event_status[confirmed] = 1
        self.event_status[aborted] = 2
        self.candidate["confirm_time_s"][confirmed] = time[:, None].expand_as(confirmed)[confirmed]
        error = self.candidate["error_heading"]
        accepted = confirmed & torch.isfinite(error).all(-1) & (error.norm(dim=-1) <= self.cfg.max_update_error_m)
        alpha = torch.where(self.samples > 0, self.cfg.ewma_alpha, 1.)
        updated = self.bias + alpha[..., None] * (error - self.bias)
        self.bias[accepted] = updated[accepted]
        self.samples += accepted.long()
        self.candidate["estimator_accepted"][confirmed] = accepted[confirmed]
        self.totals["ewma_accepted"] += accepted.sum(-1)
        self.totals["ewma_rejected"] += (confirmed & ~accepted).sum(-1)
        signed = self.candidate["error"]
        measured = confirmed & torch.isfinite(signed).all(-1)
        self.totals["signed_error_samples"] += measured.sum(-1)
        for axis, index in (("x", 0), ("y", 1)):
            self.totals[f"signed_error_{axis}_sum_m"] += torch.where(measured, signed[..., index], 0.).sum(-1)
            self.totals[f"abs_error_{axis}_sum_m"] += torch.where(measured, signed[..., index].abs(), 0.).sum(-1)
        self.totals["error_xy_sum_m"] += torch.where(measured, signed[..., :2].norm(dim=-1), 0.).sum(-1)

    def snapshot(self):
        """Copy before auto reset; a reset must not alter exported terminal data."""
        return {
            "plan_event": self.plan_event.clone(), "event_status": self.event_status.clone(),
            "plan": {name: value.clone() for name, value in self.plan.items()},
            "touchdown": {name: value.clone() for name, value in self.candidate.items()},
            "bias_heading": self.bias.clone(), "bias_samples": self.samples.clone(),
        }
