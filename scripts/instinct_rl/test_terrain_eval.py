"""CPU geometry and terminal-state checks; no Isaac Sim or policy rollout."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import eval_stairs
from terrain_eval_cases import TERRAIN_SUITE, make_terrain_cases, profile_height


def routes(num_envs=2, mode="mixed", seed=42):
    return make_terrain_cases(num_envs, seed, mode, 4, (0.08, 0.20), (0.25, 0.40),
                              2.0, True, 0.20, 0.25, 1.2)


class TerrainGeometryTests(unittest.TestCase):
    def test_seed_prefix_and_suite_coverage(self):
        self.assertEqual(routes(), routes(4)[:2])
        self.assertNotEqual(routes(), routes(seed=43))
        suite = routes(5, "suite")
        self.assertEqual([case["terrain"] for case in suite], list(TERRAIN_SUITE))
        self.assertEqual([len(case["checkpoints"]) for case in suite], [0, 1, 1, 1, 3])

    def test_profiles_are_contiguous_and_ramps_have_exact_requested_angles(self):
        for case in routes(5, "suite"):
            for a, b in zip(case["profile"], case["profile"][1:]):
                self.assertAlmostEqual(a[1], b[0])
            for left, right, low, high in case["profile"]:
                self.assertGreater(right, left)
                midpoint = (left + right) / 2
                self.assertAlmostEqual(profile_height(case, midpoint), (low + high) / 2)
                if low != high:
                    angle = math.degrees(math.atan(abs(high - low) / (right - left)))
                    self.assertAlmostEqual(angle, case["slope_angle_deg"])
            self.assertEqual(profile_height(case, case["goal_x_m"]), 0.0)

    def test_mesh_has_upward_tops_and_closed_positive_volume_prisms(self):
        import numpy as np

        for case in routes(5, "suite"):
            vertices = np.asarray(case["mesh_vertices"])
            faces = np.asarray(case["mesh_faces"])
            for i, (left, right, low, high) in enumerate(case["profile"]):
                v = vertices[8 * i:8 * (i + 1)]
                f = faces[12 * i:12 * (i + 1)] - 8 * i
                a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
                normals = np.cross(b - a, c - a)
                self.assertTrue((normals[2:4, 2] > 0).all())
                volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
                bottom = min(-0.1, low - 0.1, high - 0.1)
                expected = (right - left) * case["width_m"] * ((low + high) / 2 - bottom)
                self.assertAlmostEqual(volume, expected)
                edges = {}
                for face in f:
                    for x, y in zip(face, np.roll(face, -1)):
                        edge = tuple(sorted((int(x), int(y))))
                        edges[edge] = edges.get(edge, 0) + 1
                self.assertEqual(set(edges.values()), {2})

    def test_invalid_suite_and_nonfinite_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 5"):
            routes(4, "suite")
        for kwargs in ({"slope_angle_range": (0, 20)}, {"ramp_length": float("nan")},
                       {"curb_height_range": (0.2, 0.1)}):
            with self.assertRaises(ValueError):
                make_terrain_cases(1, 42, "mixed", 4, (.08, .2), (.25, .4), 2, True, .2, .25, 1.2, **kwargs)

    def test_new_entry_defaults_and_cli_overrides_work_without_simulator(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "preview"
            with mock.patch("sys.argv", ["eval_mixed_parkour.py", "--dry_run", "--load_run", "example",
                                          "--checkpoint", "model_22000.pt", "--terrain_mode", "suite",
                                          "--num_envs", "5", "--output_dir", str(output)]):
                eval_stairs.main({"task": "Instinct-Parkour-Mixed-Amp-G1-v0", "terrain_mode": "mixed"})
            manifest = json.loads((output / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["protocol"], "terrain_routes_v1")
            self.assertEqual(manifest["arguments"]["task"], "Instinct-Parkour-Mixed-Amp-G1-v0")
            self.assertTrue(manifest["arguments"]["video"])
            self.assertEqual(len(manifest["cases"]), 5)


class TerrainTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_stair_eval import TensorMetricTests

        TensorMetricTests.setUpClass()
        cls.base_tests = TensorMetricTests()
        cls.namespace = TensorMetricTests.namespace
        cls.torch = TensorMetricTests.torch

    def environment(self, mixed_lanes=True):
        env = self.base_tests.make_env(mode="up_down")
        cases = routes() if mixed_lanes else [routes(1, "curb")[0], routes(2, "slope")[1]]
        env.scene.terrain.terrain_generator.cases = cases
        env.scene.env_origins.zero_()
        for i, case in enumerate(cases):
            self.move(env, i, case["goal_x_m"], 0.0)
        return env, cases

    def move(self, env, i, x, height):
        data = env.scene["robot"].data
        data.root_pos_w[i] = self.torch.tensor([x, 0.0, height + 0.9])
        data.body_pos_w[i] = self.torch.tensor([[x - .1, -.1, height + .06], [x + .1, .1, height + .06]])

    def test_variable_length_profiles_match_mesh_height_and_detect_ramp_fall(self):
        env, cases = self.environment(mixed_lanes=False)
        for fraction in (0.0, .11, .23, .37, .51, .83, 1.0):
            xs = [case["lane_min_x_m"] + fraction * (case["lane_max_x_m"] - case["lane_min_x_m"])
                  for case in cases]
            actual = self.namespace["ground_height"](env, self.torch.tensor(xs)).tolist()
            for case, x, height in zip(cases, xs, actual):
                self.assertAlmostEqual(height, profile_height(case, x), places=5)
        left, right, low, high = next(p for p in cases[1]["profile"] if p[2] != p[3])
        self.move(env, 1, (left + right) / 2, (low + high) / 2)
        env.scene["robot"].data.root_pos_w[1, 2] -= .6
        self.assertEqual(self.namespace["stair_root_height"](env).tolist(), [False, True])

    def test_final_goal_requires_all_supported_checkpoints_and_resets_them(self):
        env, cases = self.environment()
        term = self.namespace["StairSuccess"](None, env)
        for _ in range(30):
            self.assertEqual(term(env).tolist(), [False, False])
        for checkpoint_id in range(3):
            for i, case in enumerate(cases):
                point = case["checkpoints"][checkpoint_id]
                self.move(env, i, (point["start_x_m"] + point["end_x_m"]) / 2, point["height_m"])
            # Airborne passage is not a supported checkpoint.
            env.scene["contact_forces"].data.net_forces_w.zero_()
            term(env)
            self.assertFalse(term.checkpoint_reached[:, checkpoint_id].any())
            env.scene["contact_forces"].data.net_forces_w[:, :, 2] = 50.0
            term(env)
            self.assertTrue(term.checkpoint_reached[:, checkpoint_id].all())
        for i, case in enumerate(cases):
            self.move(env, i, case["goal_x_m"], 0.0)
        for _ in range(25):
            result = term(env)
        self.assertEqual(result.tolist(), [True, True])
        self.assertEqual(term.snapshot["checkpoints_passed"].tolist(), [3, 3])
        term.reset([0])
        self.assertEqual(term.checkpoint_reached.sum(dim=1).tolist(), [0, 3])
        self.assertEqual(term.snapshot["checkpoints_passed"].tolist(), [3, 3])


if __name__ == "__main__":
    unittest.main()
