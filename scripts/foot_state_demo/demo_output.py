"""CSV records and spectator-only video overlays for the independent demo."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from foot_state import STATE_NAMES

TRACE_FIELDS = [
    "global_step", "env_id", "episode_index", "episode_step", "time_s", "foot", "state",
    "support_valid", "support_count", "reference_foot", "upward_force_n", "horizontal_force_n",
    "contact_score_heuristic", "touchdown", "liftoff", "done", "termination_reasons",
] + [f"sole_{axis}_b_m" for axis in "xyz"] + [f"sole_v{axis}_b_m_s" for axis in "xyz"] + [
    f"sole_{axis}_other_foot_m" for axis in "xyz"
] + ["fk_position_error_m"]
EVENT_FIELDS = [
    "global_step", "env_id", "episode_index", "foot", "event", "candidate_time_s",
    "confirmation_time_s", "confirmation_delay_s", "candidate_reference_valid",
] + [f"candidate_{axis}_other_foot_m" for axis in "xyz"]


class DemoOutput:
    """Log every terminal sample and retain only small scalar metric arrays."""

    def __init__(self, output):
        self.output = Path(output)
        self.trace_file = (self.output / "foot_trace.csv").open("x", newline="", encoding="utf-8")
        self.event_file = (self.output / "foot_events.csv").open("x", newline="", encoding="utf-8")
        self.trace = csv.DictWriter(self.trace_file, fieldnames=TRACE_FIELDS)
        self.events = csv.DictWriter(self.event_file, fieldnames=EVENT_FIELDS)
        self.trace.writeheader()
        self.events.writeheader()
        self.errors = []
        self.update_ms = []
        self.rows = 0
        self.event_counts = {"touchdown": 0, "liftoff": 0}

    def append(self, step, time_s, state, force_w, episode_index, episode_step, dones, reasons,
               active, errors=None):
        for env_id in np.flatnonzero(active):
            for foot_id, foot in enumerate(("left", "right")):
                row = {
                    "global_step": step, "env_id": env_id, "episode_index": episode_index[env_id],
                    "episode_step": episode_step[env_id], "time_s": time_s, "foot": foot,
                    "state": STATE_NAMES[state["state"][env_id, foot_id]],
                    "support_valid": int(state["support_valid"][env_id, foot_id]),
                    "support_count": state["support_count"][env_id],
                    "reference_foot": int(state["reference_foot"][env_id]),
                    "upward_force_n": max(0.0, force_w[env_id, foot_id, 2]),
                    "horizontal_force_n": np.linalg.norm(force_w[env_id, foot_id, :2]),
                    "contact_score_heuristic": state["contact_score"][env_id, foot_id],
                    "touchdown": int(state["touchdown"][env_id, foot_id]),
                    "liftoff": int(state["liftoff"][env_id, foot_id]),
                    "done": int(dones[env_id]), "termination_reasons": reasons[env_id],
                    "fk_position_error_m": "" if errors is None else errors[env_id, foot_id],
                }
                for axis_id, axis in enumerate("xyz"):
                    row[f"sole_{axis}_b_m"] = state["position_b"][env_id, foot_id, axis_id]
                    row[f"sole_v{axis}_b_m_s"] = state["velocity_b"][env_id, foot_id, axis_id]
                    row[f"sole_{axis}_other_foot_m"] = state["relative_to_other"][env_id, foot_id, axis_id]
                self.trace.writerow(row)
                self.rows += 1
                if errors is not None:
                    self.errors.append(float(errors[env_id, foot_id]))
                for event in ("touchdown", "liftoff"):
                    if not state[event][env_id, foot_id]:
                        continue
                    time_key = "candidate_time_s" if event == "touchdown" else "release_candidate_time_s"
                    candidate = state[time_key][env_id, foot_id]
                    event_row = {
                        "global_step": step, "env_id": env_id, "episode_index": episode_index[env_id],
                        "foot": foot, "event": event, "candidate_time_s": candidate,
                        "confirmation_time_s": time_s, "confirmation_delay_s": time_s - candidate,
                        "candidate_reference_valid": (
                            int(state["candidate_reference_valid"][env_id, foot_id]) if event == "touchdown" else ""
                        ),
                    }
                    for axis_id, axis in enumerate("xyz"):
                        event_row[f"candidate_{axis}_other_foot_m"] = (
                            state["candidate_relative_position"][env_id, foot_id, axis_id]
                            if event == "touchdown" else ""
                        )
                    self.events.writerow(event_row)
                    self.event_counts[event] += 1
        self.trace_file.flush()
        self.event_file.flush()

    def close(self, report):
        self.trace_file.close()
        self.event_file.close()
        report.update({
            "foot_trace_rows": self.rows, "event_counts": self.event_counts,
            "fk_position_error_m": None if not self.errors else {
                "mean": float(np.mean(self.errors)), "p95": float(np.percentile(self.errors, 95)),
                "max": float(np.max(self.errors)),
            },
            "observer_cpu_ms": None if not self.update_ms else {
                "mean": float(np.mean(self.update_ms)), "p95": float(np.percentile(self.update_ms, 95)),
            },
            "metric_scope": (
                "FK/model consistency and force-gated contact heuristic; not no-slip or balance certification"
            ),
        })
        (self.output / "foot_summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )


def overlay_frame(frame, state, env_id, step, done=False):
    """Annotate only the spectator RGB, never the policy's depth camera."""
    from PIL import Image, ImageDraw

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 spectator frame")
    image = Image.fromarray(array).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = min(490, image.width)
    height = min(220, image.height)
    draw.rectangle((0, 0, width - 1, height - 1), fill=(16, 20, 26))
    draw.text((10, 8), f"Foot state | env {env_id} | step {step}", fill="white")
    draw.text((10, 28), "Pre-reset terminal sample" if done else "Body-frame XY | forward up", fill="white")
    center_x, center_y, scale = width - 95, height - 70, min(width / 2, height) * 0.7
    draw.line((center_x, center_y + 28, center_x, center_y - 65), fill="gray")
    draw.line((center_x - 60, center_y, center_x + 60, center_y), fill="gray")
    for foot_id, (name, color) in enumerate((("L", "lime"), ("R", "cyan"))):
        p = state["position_b"][env_id, foot_id]
        v = state["velocity_b"][env_id, foot_id]
        label = STATE_NAMES[state["state"][env_id, foot_id]]
        y = 52 + 48 * foot_id
        draw.text((10, y), f"{name}: {label}", fill=color)
        draw.text((10, y + 15), f"p=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) m", fill=color)
        draw.text((10, y + 29), f"v=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})", fill=color)
        x_pixel, y_pixel = center_x - p[1] * scale, center_y - p[0] * scale
        draw.ellipse((x_pixel - 5, y_pixel - 5, x_pixel + 5, y_pixel + 5), fill=color)
    draw.text(
        (10, height - 28),
        f"Support proxy: {state['support_count'][env_id]} | ref: {state['reference_foot'][env_id]}", fill="white",
    )
    return np.asarray(image)
