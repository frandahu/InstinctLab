"""CPU contracts for route selection, geometric edge evidence and frozen errors."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from foot_state import FootStateObserver
from foothold_output import FootholdOutput
from foothold_reference import LaneGeometry, ReferenceConfig, arrange_curbs


def curb_case():
    return {"width_m": 2.0, "direction": "up_down", "terrain": "curb", "flights": [],
            "profile": [[-1, 1, 0, 0], [1, 2, .15, .15], [2, 2.18, 0, 0],
                        [2.18, 3.18, .15, .15], [3.18, 4.03, 0, 0],
                        [4.03, 5.03, .15, .15], [5.03, 7, 0, 0]],
            "curbs": [{"start_x_m": a, "end_x_m": b, "height_m": .15, "depth_m": 1., "gap_after_m": gap}
                      for a, b, gap in ((1, 2, .18), (2.18, 3.18, .85), (4.03, 5.03, 1.))]}


class GeometryTests(unittest.TestCase):
    def test_close_gap_bridges_but_far_gap_requires_ground_reference(self):
        geometry = LaneGeometry(curb_case())
        bridge = geometry.plan([1.85, .1, .15], 1)
        self.assertTrue(bridge["valid"])
        self.assertEqual(bridge["route"], "bridge_short_gap")
        self.assertGreaterEqual(bridge["target_lane_m"][0], 2.298)
        self.assertEqual(bridge["target_lane_m"][2], .15)
        descend = geometry.plan([3.03, .1, .15], 1)
        self.assertEqual(descend["route"], "descend_to_ground")
        self.assertEqual(descend["target_lane_m"][2], 0)
        ascend = geometry.plan([3.90, -.1, 0], 0)
        self.assertEqual(ascend["route"], "ground_then_ascend")
        self.assertEqual(ascend["target_lane_m"][2], .15)

    def test_short_gap_is_not_a_license_to_exceed_reach_or_height(self):
        case = curb_case()
        geometry = LaneGeometry(case, ReferenceConfig(max_step_m=.40))
        self.assertFalse(geometry.plan([1.882, .1, .15], 1)["valid"])
        case["profile"][3][2:] = [.5, .5]
        self.assertFalse(LaneGeometry(case).plan([1.882, .1, .15], 1)["valid"])

    def test_footprint_rotation_airborne_gate_and_coplanar_seams(self):
        geometry = LaneGeometry(curb_case())
        inside = geometry.footprint([1.86, 0, .15], np.eye(3))
        self.assertAlmostEqual(inside["edge_margin_m"], .047)
        self.assertEqual(geometry.footprint([1.90, 0, .15], np.eye(3))["edge_status"], "near_edge")
        overhang = geometry.footprint([1.92, 0, .15], np.eye(3))
        self.assertAlmostEqual(overhang["edge_margin_m"], -.013)
        yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(geometry.footprint([1.92, 0, .15], yaw)["edge_status"], "inside")
        heel_only = geometry.footprint([2.03, 0, .15], np.eye(3))
        self.assertTrue(heel_only["surface_near"])
        self.assertEqual(heel_only["edge_status"], "overhang")
        self.assertNotEqual(heel_only["surface_id"], heel_only["center_surface_id"])
        self.assertEqual(heel_only["surface_phase"], "curb")
        self.assertEqual(heel_only["phase"], "curb_gap")
        self.assertFalse(geometry.footprint([1.92, 0, .35], np.eye(3))["surface_near"])
        flat = curb_case()
        flat["profile"] = [[-1, 2, 0, 0], [2, 5, 0, 0]]
        self.assertEqual(LaneGeometry(flat).footprint([2, 0, 0], np.eye(3))["edge_status"], "inside")

    def test_rearranged_mesh_top_checkpoints_and_case_identity_stay_consistent(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "instinct_rl"))
        from terrain_eval_cases import make_terrain_cases, profile_height

        original = make_terrain_cases(2, 42, "mixed", 4, (.1, .2), (.28, .4), 2., True,
                                      .2, .25, 1.2, curb_count=5)
        rebuilt = arrange_curbs(original, [.18, .85, .22, 1.0])
        self.assertEqual([c["gap_after_m"] for c in rebuilt[0]["curbs"]], [.18, .85, .22, 1., 1.])
        self.assertEqual(original[0]["curbs"][0]["gap_after_m"], 1.)
        for case in rebuilt:
            geometry = LaneGeometry(case)
            for point in case["checkpoints"]:
                middle = (point["start_x_m"] + point["end_x_m"]) / 2
                self.assertAlmostEqual(profile_height(case, middle), point["height_m"])
                self.assertAlmostEqual(geometry.height(geometry.index(middle), middle), point["height_m"])
            # The actual top vertices of the last prism are on the final ground.
            self.assertTrue(all(v[2] == 0 for v in case["mesh_vertices"][-4:]))
            self.assertTrue(any(geometry.phase(sum(segment[:2])/2)[0] == "stairs_down"
                                for segment in geometry.segments))


class RecordingTests(unittest.TestCase):
    def test_candidate_latches_target_error_before_confirmation_and_keeps_terminal_frame(self):
        observer = FootStateObserver(2)
        positions = np.array([[[1.85, .1, .15], [1.6, -.1, .15]]] * 2)
        rotations = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
        rotations[:, 0] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        episode = np.array([0, 8])
        with tempfile.TemporaryDirectory() as directory:
            log = FootholdOutput(directory, [curb_case(), curb_case()], ReferenceConfig(), 20)
            def tick(step, force):
                forces = np.zeros((2, 2, 3))
                forces[..., 2] = force
                state = observer.update(step, step*.02, positions, rotations, positions*0, forces[..., 2])
                log.append(step, step*.02, state, positions, rotations, forces, episode,
                           np.array([step, step]), np.array([True, True]))
            tick(1, [0, 0])
            tick(2, [100, 0])
            tick(3, [100, 0])  # The right target must exist before its next contact.
            positions[:, 1, 0] = 2.31
            tick(4, [100, 100])
            positions[:, 1, 0] = 2.45  # Confirmation position must NOT replace first contact.
            tick(5, [100, 100])
            tick(6, [0, 100])
            tick(7, [0, 100])
            positions[:, 0, 0], positions[:, 0, 2] = 3.5, 0
            tick(8, [100, 100])  # Ground contact between far curbs.
            tick(9, [100, 100])
            tick(10, [100, 0])
            tick(11, [100, 0])
            positions[:, 1, 0] = 4.25
            tick(12, [100, 100])
            tick(13, [100, 100])
            observer.reset([0])
            log.reset([0])
            report = log.close()
            with (Path(directory) / "foothold_events.csv").open() as stream:
                events = list(csv.DictReader(stream))
            with (Path(directory) / "foothold_targets.csv").open() as stream:
                targets = list(csv.DictReader(stream))
            json.loads((Path(directory) / "foothold_summary.json").read_text())
        right = [e for e in events if e["foot"] == "right" and float(e["candidate_time_s"]) == .08]
        self.assertEqual(len(right), 2)
        self.assertEqual(right[1]["episode_index"], "8")
        self.assertAlmostEqual(float(right[0]["actual_x_lane_m"]), 2.31)
        self.assertAlmostEqual(float(right[0]["error_x_lane_m"]), .012)
        self.assertAlmostEqual(float(right[0]["error_x_frozen_support_m"]), 0)
        self.assertAlmostEqual(float(right[0]["error_y_frozen_support_m"]), -.012)
        self.assertEqual(right[0]["support_error_valid"], "1")
        self.assertLess(float(targets[0]["time_s"]), float(right[0]["candidate_time_s"]))
        self.assertEqual(report["by_phase"]["curb"]["world_error_samples"], 4)
        self.assertEqual(report["curb_rule_counts"]["bridge_short_gap/matched"], 2)
        self.assertEqual(report["curb_rule_counts"]["via_ground/matched"], 2)

    def test_lost_support_masks_support_error_but_world_error_survives(self):
        observer = FootStateObserver(1)
        p = np.array([[[1.85, .1, .15], [1.6, -.1, .15]]])
        r = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
        with tempfile.TemporaryDirectory() as directory:
            log = FootholdOutput(directory, [curb_case()], ReferenceConfig(), 20)
            for step, force in enumerate(([0, 0], [100, 0], [100, 0], [0, 0], [0, 100], [0, 100]), 1):
                if step >= 5:
                    p[0, 1, 0] = 2.31
                forces = np.zeros((1, 2, 3))
                forces[0, :, 2] = force
                state = observer.update(step, step*.02, p, r, p*0, forces[..., 2])
                log.append(step, step*.02, state, p, r, forces, np.array([0]), np.array([step]), [True])
            log.close()
            with (Path(directory) / "foothold_events.csv").open() as stream:
                event = [e for e in csv.DictReader(stream) if e["foot"] == "right"][0]
        self.assertEqual(event["world_error_valid"], "1")
        self.assertEqual(event["support_error_valid"], "0")
        self.assertEqual(event["error_x_frozen_support_m"], "")
        self.assertEqual(event["support_error_invalid_reason"], "support_proxy_lost")


if __name__ == "__main__":
    unittest.main()
