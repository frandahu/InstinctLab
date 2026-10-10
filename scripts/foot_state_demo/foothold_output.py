"""Passive foothold targets, candidate-latched errors and loaded edge evidence."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from foot_state import AIR, RELEASE, SUPPORT
from foothold_reference import LaneGeometry

XYZ = "xyz"
COMMON = ["global_step", "time_s", "env_id", "episode_index", "episode_step", "foot"]
GEOMETRY = ["surface_id", "center_surface_id", "surface_ambiguous", "phase", "curb_id", "edge_margin_m", "center_height_gap_m",
            "surface_phase", "surface_curb_id", "surface_near", "edge_status", "loaded_edge", "support_valid"]
TRACE = COMMON + GEOMETRY + ["upward_force_n", "target_valid", "target_surface_id", "target_route"]
TRACE += [f"sole_{a}_lane_m" for a in XYZ] + [f"sole_r{i}{j}_lane" for i in range(3) for j in range(3)]
TRACE += [f"target_{a}_lane_m" for a in XYZ] + [f"corner_{i}_{a}_lane_m" for i in range(4) for a in XYZ]
TARGET = COMMON + ["target_id", "valid", "reason", "route", "surface_id", "anchor_foot",
                   "privileged_world_reference"] + [f"target_{a}_lane_m" for a in XYZ]
EVENT = COMMON + GEOMETRY + ["candidate_time_s", "confirmation_time_s", "confirmation_global_step", "target_id", "target_time_s",
        "target_valid", "world_error_valid", "support_error_valid", "support_error_invalid_reason",
        "target_route", "target_surface_id", "target_surface_matched", "error_xy_m",
        "transition_from_curb", "ground_touchdown_between", "curb_rule", "curb_rule_matched"]
EVENT += [f"actual_{a}_lane_m" for a in XYZ] + [f"target_{a}_lane_m" for a in XYZ]
EVENT += [f"error_{a}_lane_m" for a in XYZ] + [f"error_{a}_frozen_support_m" for a in XYZ]


class FootholdOutput:
    """Use world truth only in this diagnostic branch; never feed the actor.

    References freeze during swing. World errors remain meaningful without a
    support proxy. Support-frame errors additionally require continuous valid
    support and bounded anchor motion, checked against privileged poses.
    """

    def __init__(self, output, cases, config, force_on_n):
        self.output, self.config = Path(output), config
        self.config_force_on = force_on_n
        self.geometry = [LaneGeometry(case, config) for case in cases]
        self.files, self.writers = [], {}
        for name, fields in (("foothold_trace", TRACE), ("foothold_targets", TARGET), ("foothold_events", EVENT)):
            stream = (self.output / f"{name}.csv").open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            self.files.append(stream)
            self.writers[name] = writer
        self.previous_state = np.zeros((len(cases), 2), dtype=int)
        self.plans, self.candidates = {}, {}
        self.last_curb = [-1] * len(cases)
        self.ground_seen = [False] * len(cases)
        self.transition_uncertain = [False] * len(cases)
        self.target_id = 0
        self.target_counts, self.edge_counts, self.rule_counts = Counter(), Counter(), Counter()
        self.invalid_targets = Counter()
        self.events = []
        self.last_display = {}

    def reset(self, env_ids):
        for env_id in env_ids:
            self.previous_state[env_id] = AIR
            self.last_curb[env_id], self.ground_seen[env_id] = -1, False
            self.transition_uncertain[env_id] = False
            for foot in range(2):
                self.plans.pop((env_id, foot), None)
                self.candidates.pop((env_id, foot), None)
                self.last_display.pop((env_id, foot), None)

    def append(self, step, time_s, state, positions, orientations, forces,
               episodes, episode_steps, active):
        for e in np.flatnonzero(active):
            geometry = self.geometry[e]
            confirmed = []
            for f, name in enumerate(("left", "right")):
                key, other = (e, f), 1-f
                common = dict(zip(COMMON, (int(step), float(time_s), int(e), int(episodes[e]), int(episode_steps[e]), name)))
                p, r, force = positions[e, f], orientations[e, f], max(0.0, float(forces[e, f, 2]))
                surface = geometry.footprint(p, r)
                edge = {k: surface[k] for k in GEOMETRY if k in surface}
                edge.update(support_valid=int(state["support_valid"][e, f]),
                            loaded_edge=int(force >= self.config_force_on and surface["surface_near"]
                                            and surface["edge_status"] in ("overhang", "near_edge")))
                if edge["loaded_edge"]:
                    self.edge_counts[f"{surface['phase']}/{surface['edge_status']}"] += 1

                # Plan as release starts, or in air before first candidate if support becomes available.
                available = state["support_valid"][e, other]
                new_release = state["state"][e, f] == RELEASE and self.previous_state[e, f] == SUPPORT
                can_plan = state["state"][e, f] == AIR or new_release
                if key not in self.plans and available and can_plan:
                    plan = geometry.plan(positions[e, other], f)
                    self.target_id += 1
                    plan.update(target_id=self.target_id, time_s=time_s, anchor_foot=other,
                                anchor_position=positions[e, other].copy(), anchor_rotation=orientations[e, other].copy(),
                                support_error_valid=True, support_error_invalid_reason="")
                    self.plans[key] = plan
                    self.target_counts[f"{'valid' if plan['valid'] else 'invalid'}/{plan.get('route', plan['reason'])}"] += 1
                    if not plan["valid"]:
                        self.invalid_targets[plan["reason"]] += 1
                    row = {**common, **{k: plan.get(k, "") for k in TARGET if k not in COMMON}}
                    for a, value in zip(XYZ, plan.get("target_lane_m", ("", "", ""))):
                        row[f"target_{a}_lane_m"] = value
                    self.writers["foothold_targets"].writerow(row)
                plan = self.plans.get(key)
                if plan is not None:
                    anchor = plan["anchor_foot"]
                    motion = np.linalg.norm(positions[e, anchor] - plan["anchor_position"])
                    rotation = orientations[e, anchor] @ plan["anchor_rotation"].T
                    angle = np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))
                    if not state["support_valid"][e, anchor]:
                        plan.update(support_error_valid=False, support_error_invalid_reason="support_proxy_lost")
                    elif motion > self.config.anchor_motion_m or angle > self.config.anchor_rotation_rad:
                        plan.update(support_error_valid=False, support_error_invalid_reason="anchor_moved_or_rotated")

                row = {**common, **edge, "upward_force_n": force, "target_valid": int(bool(plan and plan["valid"])),
                       "target_surface_id": plan.get("surface_id", "") if plan else "",
                       "target_route": plan.get("route", "") if plan else ""}
                for a, value in zip(XYZ, p):
                    row[f"sole_{a}_lane_m"] = value
                for i in range(3):
                    for j in range(3):
                        row[f"sole_r{i}{j}_lane"] = r[i, j]
                for a, value in zip(XYZ, plan.get("target_lane_m", ("", "", "")) if plan else ("", "", "")):
                    row[f"target_{a}_lane_m"] = value
                for i, corner in enumerate(surface["corners_lane_m"]):
                    for a, value in zip(XYZ, corner):
                        row[f"corner_{i}_{a}_lane_m"] = value
                self.writers["foothold_trace"].writerow(row)

                # Latch geometry AND target at the first high-force candidate, not its confirmation.
                if abs(state["candidate_time_s"][e, f] - time_s) < 1e-9:
                    event = {**common, **edge, "candidate_time_s": time_s, "target_valid": int(bool(plan and plan["valid"])),
                             "world_error_valid": int(bool(plan and plan["valid"])),
                             "support_error_valid": int(bool(plan and plan["valid"] and plan["support_error_valid"]
                                                             and state["candidate_reference_valid"][e, f])),
                             "support_error_invalid_reason": "no_target" if not plan else plan["support_error_invalid_reason"],
                             "target_id": plan["target_id"] if plan else "",
                             "target_time_s": plan["time_s"] if plan else "",
                             "target_route": plan.get("route", "") if plan else "",
                             "target_surface_id": plan.get("surface_id", "") if plan else ""}
                    if plan and not event["support_error_valid"] and not event["support_error_invalid_reason"]:
                        event["support_error_invalid_reason"] = "candidate_reference_invalid"
                    for a, value in zip(XYZ, p):
                        event[f"actual_{a}_lane_m"] = float(value)
                    if plan and plan["valid"]:
                        target = np.array(plan["target_lane_m"])
                        error = p - target
                        event.update(target_surface_matched=int(surface["surface_id"] == plan["surface_id"]),
                                     error_xy_m=float(np.linalg.norm(error[:2])))
                        for a, value, expected in zip(XYZ, error, target):
                            event[f"error_{a}_lane_m"], event[f"target_{a}_lane_m"] = float(value), float(expected)
                        if event["support_error_valid"]:
                            # Both positions in ONE frozen frame; never compare two moving frames.
                            for a, value in zip(XYZ, plan["anchor_rotation"].T @ error):
                                event[f"error_{a}_frozen_support_m"] = float(value)
                    self.candidates[key] = event
                if state["touchdown"][e, f] and key in self.candidates:
                    event = self.candidates.pop(key)
                    event.update(confirmation_global_step=int(step), confirmation_time_s=time_s)
                    confirmed.append(event)
                    self.plans.pop(key, None)
                elif state["state"][e, f] in (AIR, SUPPORT):
                    # Canceled candidates and initial standing contacts are not touchdowns.
                    self.candidates.pop(key, None)
                if state["state"][e, f] == SUPPORT and not state["touchdown"][e, f]:
                    self.plans.pop(key, None)
            # Event chronology must not depend on left/right loop ordering.
            for event in sorted(confirmed, key=lambda v: (v["candidate_time_s"], v["phase"] != "curb_gap")):
                self._curb_transition(e, event)
                self.writers["foothold_events"].writerow(event)
                self.events.append(event)
                self.last_display[e, 0 if event["foot"] == "left" else 1] = event
            self.previous_state[e] = state["state"][e]
        for stream in self.files:
            stream.flush()

    def _curb_transition(self, e, event):
        if not event["surface_near"] or event["surface_ambiguous"]:
            if self.last_curb[e] >= 0:
                self.transition_uncertain[e] = True
            return
        # A center over the gap with its heel on the upper curb is NOT ground support.
        curb_id = event["surface_curb_id"]
        if (event["phase"] == "curb_gap" and event["curb_id"] == self.last_curb[e]
                and event["surface_phase"] in ("curb_gap", "ground")):
            self.ground_seen[e] = True
        if event["surface_phase"] != "curb":
            return
        previous = self.last_curb[e]
        if previous >= 0 and curb_id > previous:
            rule = self.geometry[e].curb_rule(previous)["rule"]
            observed_ground = self.ground_seen[e]
            matched = (not observed_ground) if rule == "bridge_short_gap" else observed_ground
            ungraded = rule == "bridge_unavailable" or self.transition_uncertain[e]
            event.update(transition_from_curb=previous, ground_touchdown_between=int(observed_ground),
                         curb_rule=rule, curb_rule_matched=(
                             "" if ungraded else int(matched and curb_id == previous + 1)))
            outcome = "ungraded" if ungraded else ("matched" if event["curb_rule_matched"] else "different")
            self.rule_counts[f"{rule}/{outcome}"] += 1
        if curb_id != previous:
            self.last_curb[e], self.ground_seen[e] = curb_id, False
            self.transition_uncertain[e] = False

    def close(self, run_status="not_supplied"):
        for stream in self.files:
            stream.close()
        by_phase = defaultdict(list)
        for event in self.events:
            by_phase[event["phase"]].append(event)
        phases = {}
        for phase, events in by_phase.items():
            errors = [e["error_xy_m"] for e in events if e["world_error_valid"]]
            phases[phase] = {"touchdowns": len(events),
                "loaded_near_edge": sum(e["loaded_edge"] and e["edge_status"] == "near_edge" for e in events),
                "loaded_overhang": sum(e["loaded_edge"] and e["edge_status"] == "overhang" for e in events),
                "surface_ambiguous": sum(e["surface_ambiguous"] for e in events),
                "world_error_samples": len(errors),
                "support_error_samples": sum(e["support_error_valid"] for e in events),
                "error_xy_mean_m": float(np.mean(errors)) if errors else None,
                "error_xy_p95_m": float(np.percentile(errors, 95)) if errors else None}
        worst = sorted((e for e in self.events if e["phase"] == "stairs_down" and e["loaded_edge"]),
                       key=lambda e: e["edge_margin_m"])[:10]
        report = {"mode": "passive_reference_diagnostics", "run_status": run_status, "by_phase": phases,
            "reference_config": vars(self.config), "target_counts": dict(self.target_counts),
            "invalid_target_reasons": dict(self.invalid_targets),
            "loaded_edge_foot_samples": dict(self.edge_counts), "curb_rule_counts": dict(self.rule_counts),
            "worst_downstairs_touchdowns": [dict(e) for e in worst],
            "scope": ("Privileged geometric reference; actor receives no target. Edge flags use a nominal shoe envelope, "
                      "not measured contact points, CoP or slip. Errors are advisory deviation, not commanded tracking error.")}
        (self.output / "foothold_summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lines = ["# Foothold diagnostic report", "", "Passive references; frozen actor receives no foothold target.", "",
                 "## Downstairs touchdown evidence", "", "|env|episode|foot|candidate time (s)|margin (m)|status|",
                 "|---|---|---|---|---|---|"]
        for e in worst:
            lines.append(f"|{e['env_id']}|{e['episode_index']}|{e['foot']}|{e['candidate_time_s']:.3f}"
                         f"|{e['edge_margin_m']:.4f}|{e['edge_status']}|")
        if not worst:
            lines += ["", "No loaded edge events recorded at downstairs touchdowns; inspect by_phase for sample count."]
        lines += ["", "Negative margin: projected shoe envelope overhang. 0 to configured edge margin: near edge.",
                  "This is geometric evidence, not a contact-pressure or stability certificate.",
                  "Time is global simulation time; video may end early or be subsampled. All envs are logged."]
        (self.output / "foothold_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def overlay(self, frame, env_id):
        """Add fixed-world target/error text to the spectator RGB only."""
        from PIL import Image, ImageDraw

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        top = max(225, image.height - 110)
        if top >= image.height:
            return frame
        draw.rectangle((0, top, min(800, image.width)-1, image.height-1), fill=(16, 20, 26))
        draw.text((10, top+5), "PASSIVE reference | actor receives no foothold targets | lane/world axes", fill="white")
        for f, name in enumerate(("L", "R")):
            plan = self.plans.get((env_id, f))
            text = f"{name}: "
            if plan and plan["valid"]:
                x, y, z = plan["target_lane_m"]
                text += f"target=({x:.2f},{y:.2f},{z:.2f}) {plan['route']}"
            else:
                text += "no active valid target"
            event = self.last_display.get((env_id, f))
            if event:
                margin = event["edge_margin_m"]
                text += f" | last {event['phase']} margin={margin:+.3f}m" if margin is not None else " | outside lane"
            draw.text((10, top+29+f*25), text, fill="lime" if f == 0 else "cyan")
        return np.asarray(image)
