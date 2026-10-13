"""Event CSVs and sample-aware reference response summaries, without Isaac imports."""

import csv
import math
import statistics
from pathlib import Path


PLAN_SCALARS = (
    "reference_id", "slot", "foot", "anchor", "plan_time_s", "bias_samples",
    "correction_requested", "projection_changed", "correction_erased", "correction_opposed",
)
POSITION_NAMES = ("planned", "requested", "applied")
PLAN_FIELDS = ["global_step", "env_id", "episode", "frame"] + list(PLAN_SCALARS)
PLAN_FIELDS += [f"{name}_{axis}_a_m" for name in POSITION_NAMES for axis in "xyz"]
PLAN_FIELDS += ["bias_heading_x_m", "bias_heading_y_m"]
TOUCHDOWN_FIELDS = PLAN_FIELDS + ["status", "contact_time_s", "confirm_time_s", "estimator_accepted"]
TOUCHDOWN_FIELDS += [f"{name}_{axis}_a_m" for name in ("actual", "error") for axis in "xyz"]
TOUCHDOWN_FIELDS += ["error_heading_x_m", "error_heading_y_m"]
TOUCHDOWN_FIELDS += ["ewma_heading_x_after_m", "ewma_heading_y_after_m", "ewma_samples_after"]


def _row(values, index, tick, env_id, episode, frame):
    result = dict(global_step=tick, env_id=env_id, episode=int(episode), frame=frame)
    result.update({name: values[name][index] for name in PLAN_SCALARS})
    for name in POSITION_NAMES:
        result.update({f"{name}_{axis}_a_m": values[name][index][i] for i, axis in enumerate("xyz")})
    result.update({f"bias_heading_{axis}_m": values["bias_heading"][index][i] for i, axis in enumerate("xy")})
    return result


class ReferenceRecorder:
    """Consume pre-reset snapshots; only confirmed contacts enter error statistics."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.plans, self.touchdowns = [], []
        self.streams = []

    def __enter__(self):
        try:
            for name, fields in (("reference_plans.csv", PLAN_FIELDS), ("touchdown_events.csv", TOUCHDOWN_FIELDS)):
                stream = (self.directory / name).open("x", newline="", encoding="utf-8")
                self.streams.append(stream)
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                setattr(self, "plan_writer" if name.startswith("reference") else "touchdown_writer", writer)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for stream in self.streams:
            stream.close()

    def record(self, tick, episodes, snapshot, done):
        # Batched transfers avoid a GPU synchronization for each CSV scalar.
        plans = {name: value.cpu().tolist() for name, value in snapshot["plan"].items()}
        contacts = {name: value.cpu().tolist() for name, value in snapshot["touchdown"].items()}
        for e, generated in enumerate(snapshot["plan_event"].cpu().tolist()):
            if generated:
                row = _row(plans, e, tick, e, episodes[e], "planning_anchor")
                self.plan_writer.writerow(row)
                self.plans.append(row)
        status = snapshot["event_status"].cpu().tolist()
        pending = snapshot["pending"].cpu().tolist()
        biases = snapshot["bias_heading"].cpu().tolist()
        samples = snapshot["bias_samples"].cpu().tolist()
        for e, feet in enumerate(status):
            for foot, event in enumerate(feet):
                label = {1: "confirmed", 2: "aborted"}.get(event)
                if label is None and done[e] and pending[e][foot]:
                    label = "reset_unconfirmed"
                if label is None:
                    continue
                values = {name: value[e] for name, value in contacts.items()}
                row = _row(values, foot, tick, e, episodes[e], "first_contact_anchor")
                row.update(status=label, contact_time_s=values["contact_time_s"][foot],
                           confirm_time_s=values["confirm_time_s"][foot] if event == 1 else "",
                           estimator_accepted=values["estimator_accepted"][foot] if event == 1 else False)
                for name in ("actual", "error"):
                    row.update({f"{name}_{axis}_a_m": values[name][foot][i] for i, axis in enumerate("xyz")})
                row.update({f"error_heading_{axis}_m": values["error_heading"][foot][i] for i, axis in enumerate("xy")})
                row.update({f"ewma_heading_{axis}_after_m": biases[e][foot][i] for i, axis in enumerate("xy")})
                row["ewma_samples_after"] = samples[e][foot]
                self.touchdown_writer.writerow(row)
                self.touchdowns.append(row)

    def summary(self):
        return summarize_reference(self.plans, self.touchdowns)


def _true(value):
    return str(value).lower() in ("true", "1")


def _mean(values):
    return statistics.mean(values) if values else None


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] * (hi - index) + values[hi] * (index - lo) if hi != lo else values[lo]


def _delta(row, end, start):
    return [float(row[f"{end}_{a}_a_m"]) - float(row[f"{start}_{a}_a_m"]) for a in "xy"]


def _norm(v):
    return math.hypot(*v)


def _error_summary(rows):
    errors = [[float(row[f"error_{a}_a_m"]) for a in "xy"] for row in rows]
    norms = [_norm(v) for v in errors]
    applied = [_delta(row, "applied", "planned") for row in rows]
    landed = [_delta(row, "actual", "planned") for row in rows]
    effective = [(a, l) for a, l in zip(applied, landed) if _norm(a) > 0.001]
    energy = sum(sum(v*v for v in a) for a, _ in effective)
    return {
        "samples": len(rows),
        "signed_error_xy_mean_m": [_mean([v[i] for v in errors]) for i in range(2)],
        "abs_error_xy_mean_m": [_mean([abs(v[i]) for v in errors]) for i in range(2)],
        "horizontal_error_mean_m": _mean(norms),
        "horizontal_error_p95_m": _percentile(norms, 0.95),
        "horizontal_error_rmse_m": math.sqrt(statistics.mean(v*v for v in norms)) if norms else None,
        "applied_offset_xy_mean_m": [_mean([v[i] for v in applied]) for i in range(2)],
        "landing_offset_xy_mean_m": [_mean([v[i] for v in landed]) for i in range(2)],
        "effective_correction_touchdowns": len(effective),
        "response_gain_proxy": sum(sum(x*y for x, y in zip(a, l)) for a, l in effective) / energy if energy else None,
        "ewma_accepted": sum(_true(row["estimator_accepted"]) for row in rows),
    }


def summarize_reference(plans, events):
    confirmed = [row for row in events if row["status"] == "confirmed"]
    requested = sum(_true(row["correction_requested"]) for row in plans)
    erased = sum(_true(row["correction_erased"]) for row in plans)
    def key(row):
        return int(row["env_id"]), int(row["episode"]), int(row["reference_id"])

    completed = {key(row) for row in confirmed}
    return {
        "reference_commands": len(plans),
        "commands_with_confirmed_touchdown": len(completed),
        "commands_without_confirmed_touchdown": sum(key(row) not in completed for row in plans),
        "corrections_requested": requested,
        "projection_changed": sum(_true(row["projection_changed"]) for row in plans),
        "corrections_erased": erased,
        "corrections_opposed": sum(_true(row["correction_opposed"]) for row in plans),
        "correction_erased_fraction": erased / requested if requested else None,
        "requested_offset_norm_mean_m": _mean([_norm(_delta(r, "requested", "planned")) for r in plans]),
        "applied_offset_norm_mean_m": _mean([_norm(_delta(r, "applied", "planned")) for r in plans]),
        "projection_delta_norm_mean_m": _mean([math.sqrt(sum(
            (float(r[f"applied_{a}_a_m"]) - float(r[f"requested_{a}_a_m"]))**2 for a in "xyz"
        )) for r in plans]),
        "aborted_candidates": sum(row["status"] == "aborted" for row in events),
        "reset_unconfirmed_candidates": sum(row["status"] == "reset_unconfirmed" for row in events),
        "confirmed": _error_summary(confirmed),
        "per_foot": {name: _error_summary([r for r in confirmed if int(r["foot"]) == foot])
                     for foot, name in enumerate(("left", "right"))},
        "response_interpretation": "response_gain_proxy includes execution bias and is descriptive, not causal tracking proof; "
                                   "compare matched baseline and +/- perturbation runs; erased proposals do not test the actor",
        "measurement": "shoe FK at first contact, in the reliable contact-anchor frame; not independent world-pose ground truth",
        "incomplete_interpretation": "commands_without_confirmed_touchdown includes invalidated, terminated and rollout-censored commands",
    }
