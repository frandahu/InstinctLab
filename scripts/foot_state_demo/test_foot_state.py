"""CPU contracts: coordinate conventions, velocity, contact timing and reset isolation."""

import csv
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from demo_output import DemoOutput
from foot_state import (AIR, CANDIDATE, RELEASE, SUPPORT, FootStateObserver,
                        LegKinematics, ObserverConfig, quaternion_matrix)


class KinematicsTests(unittest.TestCase):
    def test_named_joint_mapping_origin_rotation_fixed_sole_and_velocity(self):
        # Analytic fixture: left radius=1 rotates to (-1, 0); right slides on X.
        urdf = '''<robot name="analytic"><link name="base"/><link name="hinge"/>
        <link name="left_ankle_roll_link"/><link name="right_ankle_roll_link"/>
        <joint name="turn" type="revolute"><parent link="base"/><child link="hinge"/>
        <origin xyz="0 .5 0" rpy="0 0 1.5707963267948966"/><axis xyz="0 0 1"/></joint>
        <joint name="sole_fixed" type="fixed"><parent link="hinge"/><child link="left_ankle_roll_link"/>
        <origin xyz=".2 0 0"/></joint>
        <joint name="slide" type="prismatic"><parent link="base"/><child link="right_ankle_roll_link"/>
        <origin xyz="0 2 0"/><axis xyz="1 0 0"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analytic.urdf"
            path.write_text(urdf, encoding="utf-8")
            model = LegKinematics(path, ["slide", "turn"], sole_offset=(.8, 0, 0))
            p, _, v = model.compute([[.3, math.pi / 2]], [[.4, 2]])
        np.testing.assert_allclose(p[0], [[-1, .5, 0], [1.1, 2, 0]], atol=1e-12)
        np.testing.assert_allclose(v[0], [[0, -2, 0], [.4, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(quaternion_matrix([[math.sqrt(.5), 0, 0, math.sqrt(.5)]])[0] @ [1, 0, 0],
                                   [0, 1, 0], atol=1e-12)

    def test_actual_g1_analytic_velocity_matches_position_derivative(self):
        from offline_demo import DEFAULT_URDF

        names = [node.attrib["name"] for node in ET.parse(DEFAULT_URDF).getroot().findall("joint")
                 if node.attrib["type"] != "fixed"]
        model = LegKinematics(DEFAULT_URDF, names)
        rng = np.random.RandomState(7)
        q = rng.uniform(-.25, .25, (2, len(names)))
        dq = rng.uniform(-1, 1, q.shape)
        _, _, velocity = model.compute(q, dq)
        epsilon = 1e-6
        plus = model.compute(q + epsilon * dq, dq)[0]
        minus = model.compute(q - epsilon * dq, dq)[0]
        np.testing.assert_allclose(velocity, (plus - minus) / (2 * epsilon), atol=1e-8)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.observer = FootStateObserver(2)
        self.p = np.array([[[0, .1, -.8], [.2, -.1, -.8]]] * 2)
        self.r = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
        self.v = np.ones((2, 2, 3))
        self.step = 0

    def update(self, force, position=None):
        self.step += 1
        return self.observer.update(self.step, self.step * .02, self.p if position is None else position,
                                    self.r, self.v, np.broadcast_to(force, (2, 2)))

    def test_bootstrap_bounce_and_touchdown_candidate_latching(self):
        self.update(100)
        state = self.update(100)
        self.assertTrue((state["state"] == SUPPORT).all())
        self.assertFalse(state["touchdown"].any())  # Already standing is not a new step.
        self.assertTrue((state["velocity_b"] == 1).all())  # Never zero stance velocity in body frame.
        self.assertTrue((self.update(0)["state"] == RELEASE).all())
        self.assertTrue(self.update(0)["liftoff"].all())
        self.assertTrue((self.update(100)["state"] == CANDIDATE).all())
        self.assertTrue((self.update(0)["state"] == AIR).all())  # One-frame collision rejected.
        self.update(100)
        p = self.p.copy()
        p[:, 1, 0] = .7
        state = self.update(100, p)
        self.assertTrue(state["touchdown"].all())
        np.testing.assert_allclose(state["candidate_relative_position"][0, 1], [.2, -.2, 0])
        np.testing.assert_allclose(state["candidate_time_s"], .14)

    def test_partial_reset_duplicate_reads_and_reference_switch(self):
        self.update(100)
        state = self.update(100)
        self.assertEqual(state["reference_foot"].tolist(), [0, 0])
        duplicate = self.observer.update(self.step, 999, self.p, self.r, self.v, np.zeros((2, 2)))
        self.assertIs(state, duplicate)
        self.observer.reset([0])
        self.v[0] = 7
        state = self.update(100)
        self.assertTrue((state["state"][0] == CANDIDATE).all())
        self.assertTrue((state["state"][1] == SUPPORT).all())
        np.testing.assert_allclose(state["velocity_b"][0], 7)
        state = self.update([[100, 100], [0, 100]])
        self.assertEqual(state["reference_foot"].tolist(), [0, 1])
        state = self.update(0)
        self.assertEqual(state["reference_foot"].tolist(), [-1, -1])

    def test_velocity_lowpass_and_invalid_measurements(self):
        self.update(0)
        self.v[:] = 0
        state = self.update(0)
        np.testing.assert_allclose(state["velocity_b"], math.exp(-2 * math.pi * 10 * .02))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.observer.update(3, .06, self.p * float("nan"), self.r, self.v, np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            ObserverConfig(force_on_n=10, force_off_n=10)

    def test_terminal_output_is_written_before_observer_reset(self):
        self.update(0)
        self.update(100)
        state = self.update(100)
        with tempfile.TemporaryDirectory() as directory:
            log = DemoOutput(directory)
            log.append(3, .06, state, np.zeros((2, 2, 3)), np.array([4, 7]), np.array([9, 11]),
                       np.array([True, False]), ["root_height", ""], np.array([True, False]),
                       errors=np.array([[.01, .03], [.5, .5]]))
            self.observer.reset([0])
            log.close({"status": "test"})
            with (Path(directory) / "foot_trace.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            with (Path(directory) / "foot_events.csv").open(encoding="utf-8") as stream:
                events = list(csv.DictReader(stream))
            report = json.loads((Path(directory) / "foot_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode_index"], "4")
        self.assertEqual(rows[0]["done"], "1")
        self.assertEqual(rows[0]["state"], "support")
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(float(events[0]["confirmation_delay_s"]), .02)
        self.assertAlmostEqual(report["fk_position_error_m"]["mean"], .02)


if __name__ == "__main__":
    unittest.main()
