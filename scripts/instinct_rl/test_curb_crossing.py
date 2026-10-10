"""CPU contracts for curb stepping; no Isaac Sim/GPU validation."""

import ast
import copy
import importlib.util
import math
import numpy as np
import tempfile
import torch
import unittest
import yaml
from pathlib import Path
from types import SimpleNamespace

from curb_crossing_warm_start import warm_start
from prepare_curb_crossing_motions import bridge_windows

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "source/instinctlab/instinctlab/tasks/parkour/config/g1"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, CFG / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load_module("curb_crossing_cases")
mdp = load_module("curb_crossing_mdp")


class Scene(dict):
    @property
    def terrain(self):
        return self["terrain"]


def environment():
    grid = geometry.curb_crossing_cases(42, 8, 4)
    origins = torch.tensor([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 6.0, 0.0]])
    data = SimpleNamespace(
        body_pos_w=torch.zeros(3, 2, 3),
        body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * 3),
        root_pos_w=origins + torch.tensor([0.0, 0.0, 0.9]),
        root_lin_vel_b=torch.zeros(3, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * 3),
    )
    # Articulation and sensor ordering intentionally differ.
    robot = SimpleNamespace(data=data, find_bodies=lambda name: ([1 if name.startswith("left") else 0], [name]))
    sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=torch.tensor([[[0.0, 0.0, 50.0]] * 2] * 3)),
        find_bodies=lambda name: ([0 if name.startswith("left") else 1], [name]),
    )
    terrain = SimpleNamespace(
        terrain_generator=SimpleNamespace(grid_cases=grid),
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_cols=4, num_rows=8)),
        terrain_levels=torch.zeros(3, dtype=torch.long),
        terrain_types=torch.tensor([1, 2, 0]),
    )
    updates = []
    terrain.update_env_origins = lambda ids, up, down: updates.append((up.tolist(), down.tolist()))
    scene = Scene(robot=robot, contact_forces=sensor, terrain=terrain)
    scene.env_origins = origins
    flags = {
        name: torch.zeros(3, dtype=torch.bool)
        for name in ("curb_goal", "root_height", "off_course", "bad_orientation", "base_contact", "nonfinite_state")
    }
    metrics = {
        name: torch.zeros(3)
        for name in ("curb_clean_crossings", "curb_ground_touched_short_gaps", "curb_route_success")
    }
    env = SimpleNamespace(
        scene=scene,
        num_envs=3,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        episode_length_buf=torch.ones(3, dtype=torch.long),
        termination_manager=SimpleNamespace(get_term=lambda name: flags[name]),
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(metrics=metrics)),
    )
    return env, grid, flags, updates, metrics


def place(env, grid, platform=None, gap=None, end=False, edge=False):
    for e, col in enumerate((1, 2, 0)):
        case = grid[0][col]
        if end:
            x, z = case["goal_x_m"], 0.0
            env.scene["robot"].data.root_pos_w[e] = env.scene.env_origins[e] + torch.tensor([x, 0.0, 0.9])
        elif gap is not None:
            c = case["curbs"][gap]
            x, z = c["end_x_m"] + c["gap_after_m"] / 2, 0.0
        else:
            c = case["curbs"][platform]
            x = c["end_x_m"] - 0.01 if edge else (c["start_x_m"] + c["end_x_m"]) / 2
            z = c["height_m"]
        for foot, body_id in enumerate((1, 0)):
            sole = torch.tensor([x, 0.1 if foot == 0 else -0.1, z])
            env.scene["robot"].data.body_pos_w[e, body_id] = (
                env.scene.env_origins[e] + sole - torch.tensor([0.039, 0.0, -0.058])
            )


def advance(env, count=3):
    events = torch.zeros(env.num_envs)
    for _ in range(count):
        env.common_step_counter += 1
        events += mdp.curb_bridge_reward(env) * env.step_dt
        mdp.curb_crossing_goal(env)  # Repeated consumers must not advance timers twice.
    return events


class CurbCrossingTests(unittest.TestCase):
    def test_collision_grid_has_platform_tops_low_gaps_and_world_targets(self):
        import trimesh

        namespace = {
            "np": np,
            "torch": torch,
            "trimesh": trimesh,
            "copy": copy,
            "curb_crossing_cases": geometry.curb_crossing_cases,
        }
        # Execute the real mesh constructors without importing Isaac Sim. This
        # verifies their collision output, not their source spelling or mocks.
        for file_name, class_name in (
            ("stair_eval_cfg.py", "StaircaseGenerator"),
            ("curb_crossing_cfg.py", "CurbCrossingGenerator"),
        ):
            tree = ast.parse((CFG / file_name).read_text(encoding="utf-8"))
            node = next(node for node in tree.body if getattr(node, "name", None) == class_name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), file_name, "exec"), namespace)
        generated = namespace["CurbCrossingGenerator"](
            SimpleNamespace(seed=42, num_rows=8, num_cols=4, size=(14.0, 4.0), gap_range=(0.18, 0.30), max_step_m=0.60),
            "cpu",
        )
        self.assertEqual(generated.terrain_origins.shape, (8, 4, 3))
        self.assertTrue(generated.terrain_mesh.is_watertight)
        centers = generated.terrain_mesh.triangles_center
        upward = generated.terrain_mesh.face_normals[:, 2] > 0.99
        for row, col in ((0, 0), (7, 1)):
            case = generated.grid_cases[row][col]
            origin = generated.terrain_origins[row, col]
            goal = generated.flat_patches["target"][row, col, 0].numpy() - origin
            np.testing.assert_allclose(goal, [case["goal_x_m"], 0.0, 0.0], atol=1e-5)
            lane = upward & (abs(centers[:, 1] - origin[1]) < 0.99)
            local_x = centers[:, 0] - origin[0]
            lane &= (local_x > -1.2) & (local_x < case["lane_max_x_m"])
            for center in centers[lane] - origin:
                roof = next((c["height_m"] for c in case["curbs"] if c["start_x_m"] <= center[0] < c["end_x_m"]), 0.0)
                self.assertAlmostEqual(center[2], roof, places=5)

    def test_curriculum_reachable_spans_flat_retention_and_seed(self):
        grid = geometry.curb_crossing_cases(42, 8, 4)
        self.assertEqual(grid, geometry.curb_crossing_cases(42, 8, 4))
        self.assertNotEqual(grid, geometry.curb_crossing_cases(43, 8, 4))
        for row in grid:
            for case in row:
                self.assertLess(case["lane_max_x_m"] + 1.2, 14.0)
                self.assertEqual(sum(case["bridge_required"]), 0 if case["terrain"] == "flat" else 3)
                for i, required in enumerate(case["bridge_required"]):
                    if required:
                        gap = case["curbs"][i]["gap_after_m"]
                        self.assertLessEqual(math.hypot(gap + 0.236, 0.2), 0.6)
        with self.assertRaisesRegex(ValueError, "envelope"):
            geometry.curb_crossing_cases(42, 8, 4, gap_range=(0.18, 0.60))

    def test_support_confirmation_clean_reward_no_farming_and_reset_isolation(self):
        env, grid, _, _, metrics = environment()
        place(env, grid, platform=0)
        advance(env)
        place(env, grid, gap=0)
        # Only lane 1 touches the gap; lane 0 is in swing with no force.
        env.scene["contact_forces"].data.net_forces_w[0] = 0
        advance(env, 1)
        self.assertEqual(mdp.curb_gap_ground_contact(env).tolist(), [0.0, 2.0, 0.0])
        env.scene["contact_forces"].data.net_forces_w[:, :, 2] = 50
        place(env, grid, platform=1, edge=True)
        self.assertEqual(advance(env, 4).tolist(), [0.0, 0.0, 0.0])
        place(env, grid, platform=1)
        self.assertEqual(advance(env, 2).tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(advance(env, 1).tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(advance(env, 4).tolist(), [0.0, 0.0, 0.0])
        state = mdp.crossing_state(env)
        mdp.reset_curb_crossing(env, torch.tensor([0]))
        self.assertEqual(metrics["curb_clean_crossings"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(state.last_platform.tolist(), [-1, 1, 1])
        self.assertTrue(state.ground_seen[1, 0])
        self.assertFalse(state.passed[0].any())

    def test_goal_requires_clean_bridges_and_final_stable_support_then_curriculum(self):
        env, grid, flags, updates, _ = environment()
        place(env, grid, platform=0)
        advance(env)
        place(env, grid, gap=0)
        env.scene["contact_forces"].data.net_forces_w[0] = 0
        advance(env, 1)
        env.scene["contact_forces"].data.net_forces_w[:, :, 2] = 50
        place(env, grid, platform=1)
        advance(env)
        place(env, grid, gap=1)  # Long-gap ground touch must not invalidate success.
        advance(env, 1)
        for platform in (2, 3, 4):
            place(env, grid, platform=platform)
            advance(env)
        place(env, grid, end=True)
        advance(env, 24)
        self.assertEqual(mdp.curb_crossing_goal(env).tolist(), [False, False, False])
        advance(env, 1)
        self.assertEqual(mdp.curb_crossing_goal(env).tolist(), [True, False, True])
        flags["curb_goal"][:] = mdp.curb_crossing_goal(env)
        flags["base_contact"][2] = True
        mdp.curb_crossing_curriculum(env, torch.arange(3))
        self.assertEqual(updates[-1], ([True, False, False], [False, True, True]))

    def test_kinematic_selection_excludes_ground_stance_inside_crossing(self):
        sole = np.zeros((200, 2, 3))
        sole[:, 0, 1], sole[:, 1, 1] = 0.1, -0.1
        sole[:30, 1, 0] = -0.2
        sole[30:50, 0, 2] = np.linspace(0.0, 0.15, 20)
        sole[50:, 0, 2] = 0.15
        sole[30:50, 1] = np.linspace([-0.2, -0.1, 0.0], [-0.2, -0.1, 0.15], 20)
        sole[50:70, 1] = [-0.2, -0.1, 0.15]
        sole[70:100, 1] = np.linspace([-0.2, -0.1, 0.15], [0.5, -0.1, 0.15], 30)
        sole[100:, 1] = [0.5, -0.1, 0.15]
        self.assertTrue(bridge_windows(sole, 50.0))
        sole[60:80, 1] = [0.2, -0.1, 0.0]
        self.assertFalse(bridge_windows(sole, 50.0))

    def test_warm_start_copies_weights_leaves_amp_optimizer_and_iteration_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "params").mkdir()
            cfg = {"policy": {"class_name": "Linear"}, "normalizers": {}, "empirical_normalization": False}
            (root / "params/agent.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            source, target = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
            torch.nn.init.constant_(source.weight, 0.7)
            torch.nn.init.constant_(source.bias, 0.2)
            checkpoint = root / "model_24000.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "iter": 24000,
                    "discriminator_state_dict": {"sentinel": 9},
                    "optimizer_state_dict": {"sentinel": 8},
                },
                checkpoint,
            )
            runner = SimpleNamespace(
                alg=SimpleNamespace(actor_critic=target, discriminator="fresh", optimizer="fresh"),
                normalizers={},
                current_learning_iteration=0,
            )
            report = warm_start(runner, checkpoint, SimpleNamespace(to_dict=lambda: cfg))
            torch.testing.assert_close(target(torch.tensor([[1.0, 2.0]])), torch.tensor([[2.3]]))
            self.assertEqual(
                (runner.alg.discriminator, runner.alg.optimizer, runner.current_learning_iteration),
                ("fresh", "fresh", 0),
            )
            self.assertEqual(report["source_iteration"], 24000)
            incompatible = dict(cfg, policy={"class_name": "Other"})
            with self.assertRaisesRegex(ValueError, "architecture"):
                warm_start(runner, checkpoint, SimpleNamespace(to_dict=lambda: incompatible))


if __name__ == "__main__":
    unittest.main()
