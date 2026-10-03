"""CPU geometry/reward/curriculum checks; not Isaac Sim integration tests."""

import ast
import builtins
import copy
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from stair_training_runtime import save_runtime_sources

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "source/instinctlab/instinctlab/tasks/parkour/config/g1"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


GEOMETRY = load_definitions(CFG / "stair_training_cases.py", {"training_cases"}, {"math": math, "random": random})
training_cases = GEOMETRY["training_cases"]
FUNCTIONS = {"route_data", "training_stair_root_height", "training_stair_off_course",
             "training_stair_goal", "stair_progress_curriculum", "moving_feet_air_time", "velocity_deficit"}
LOGIC = load_definitions(CFG / "stair_training_cfg.py", FUNCTIONS, {"torch": torch, "math": math})


def environment(levels=(0, 4), columns=(0, 1)):
    n = len(levels)
    grid = training_cases(42, 8, 16)
    origins = torch.tensor([[row * 12.0, col * 4.0, 0.0] for row, col in zip(levels, columns)])
    data = SimpleNamespace(
        root_pos_w=origins + torch.tensor([0.0, 0.0, 0.85]),
        root_lin_vel_b=torch.zeros(n, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]] * n),
        body_pos_w=origins[:, None, :] + torch.tensor([[[0., .15, .058], [0., -.15, .058]]]),
    )
    robot = SimpleNamespace(data=data, find_bodies=lambda pattern: ([0, 1], ["left", "right"]))
    flags = {name: torch.zeros(n, dtype=torch.bool) for name in (
        "stair_goal", "root_height", "off_course", "bad_orientation", "base_contact", "nonfinite_state"
    )}
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_cols=16)),
        terrain_generator=SimpleNamespace(grid_cases=grid),
        terrain_levels=torch.tensor(levels), terrain_types=torch.tensor(columns),
    )
    updates = []

    def update(ids, up, down):
        updates.append((up.clone(), down.clone()))
        terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + up.long() - down.long()).clamp(0, 7)

    terrain.update_env_origins = update

    class Scene(dict):
        pass

    forces = torch.zeros(n, 2, 3)
    forces[:, :, 2] = 50.
    contact_forces = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=forces), find_bodies=lambda pattern: ([0, 1], ["left", "right"])
    )
    scene = Scene(robot=robot, terrain=terrain, contact_forces=contact_forces)
    scene.terrain = terrain
    scene.env_origins = origins
    command = torch.tensor([[.5, 0., 0.]] * n)
    env = SimpleNamespace(
        device="cpu", num_envs=n, step_dt=.02, scene=scene,
        episode_length_buf=torch.ones(n, dtype=torch.long) * 100,
        termination_manager=SimpleNamespace(get_term=lambda name: flags[name]),
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    return env, flags, updates, command


class GeometryTests(unittest.TestCase):
    def test_reproducible_balanced_sparse_stairs_and_flat_first_row(self):
        grid = training_cases(42, 8, 16)
        self.assertEqual(grid, training_cases(42, 8, 16))
        self.assertNotEqual(grid, training_cases(43, 8, 16))
        for row, lanes in enumerate(grid):
            for case in lanes:
                self.assertAlmostEqual(sum(case["riser_deltas_m"]), 0.0)
                self.assertEqual(case["surface_heights_m"][-1], 0.0)
                self.assertLess(case["lane_max_x_m"] + 1.20, 12.0)
                self.assertTrue(all(x > 0 for x in case["tread_depths_m"]))
                self.assertLess(max(case["surface_heights_m"]), 1.4)
                if row == 0:
                    self.assertEqual(case["direction"], "flat")
                    self.assertFalse(any(case["surface_heights_m"]))
                else:
                    self.assertTrue(all(x > 0 for x in case["riser_deltas_m"][:6]))
                    self.assertEqual(case["riser_deltas_m"][6], 0.0)
                    self.assertTrue(all(x < 0 for x in case["riser_deltas_m"][7:]))
                if row >= 3:
                    for flight in (case["riser_deltas_m"][:6], [-x for x in case["riser_deltas_m"][7:]]):
                        changed = sum(x != case["nominal_riser_height_m"] for x in flight)
                        self.assertEqual(changed, 2)
        heights = [row[0]["nominal_riser_height_m"] for row in grid]
        self.assertEqual(heights, sorted(heights))
        self.assertAlmostEqual(heights[-1], .18)

    def test_real_collision_mesh_grid_targets_and_surface_heights(self):
        import trimesh

        namespace = {"torch": torch, "np": np, "trimesh": trimesh, "copy": copy,
                     "training_cases": training_cases}
        load_definitions(CFG / "stair_eval_cfg.py", {"StaircaseGenerator"}, namespace)
        load_definitions(CFG / "stair_training_cfg.py", {"TrainingStaircaseGenerator"}, namespace)
        cfg = SimpleNamespace(seed=42, num_rows=8, num_cols=16, size=(12.0, 4.0))
        generated = namespace["TrainingStaircaseGenerator"](cfg, "cpu")
        self.assertEqual(generated.terrain_origins.shape, (8, 16, 3))
        self.assertEqual(tuple(generated.flat_patches["target"].shape), (8, 16, 1, 3))
        self.assertTrue(generated.terrain_mesh.is_watertight)
        for row, col in ((0, 0), (3, 7), (7, 15)):
            origin = generated.terrain_origins[row, col]
            goal = generated.flat_patches["target"][row, col, 0].numpy()
            case = generated.grid_cases[row][col]
            self.assertAlmostEqual(goal[0] - origin[0], case["goal_x_m"], places=4)
            self.assertAlmostEqual(goal[1], origin[1])
            # Each horizontal upward-facing triangle belongs to its expected
            # slab. Validate actual collision top elevations, without ray deps.
            centers = generated.terrain_mesh.triangles_center
            up = generated.terrain_mesh.face_normals[:, 2] > .99
            lane = (abs(centers[:, 1] - origin[1]) < .99) & (centers[:, 0] > origin[0] - 1.2)
            lane &= centers[:, 0] < origin[0] + case["lane_max_x_m"]
            for center in centers[up & lane]:
                edge, z = case["stair_start_x_m"], 0.0
                x = center[0] - origin[0]
                if x >= edge:
                    for depth, height in zip(case["tread_depths_m"], case["surface_heights_m"]):
                        edge += depth
                        if x < edge:
                            z = height
                            break
                self.assertAlmostEqual(center[2], z, places=5)


class RewardAndCurriculumTests(unittest.TestCase):
    def test_ground_relative_fall_and_lookup_after_level_change(self):
        env, _, _, _ = environment()
        row = env.scene.terrain.terrain_generator.grid_cases[4][1]
        summit_x = row["stair_start_x_m"] + sum(row["tread_depths_m"][:6]) + .6
        summit_z = row["surface_heights_m"][6]
        env.scene["robot"].data.root_pos_w[1] = env.scene.env_origins[1] + torch.tensor([summit_x, 0., summit_z + .85])
        self.assertEqual(LOGIC["training_stair_root_height"](env).tolist(), [False, False])
        env.scene["robot"].data.root_pos_w[1, 2] = summit_z + .4
        self.assertEqual(LOGIC["training_stair_root_height"](env).tolist(), [False, True])
        env.scene.terrain.terrain_levels[:] = 0
        self.assertEqual(LOGIC["training_stair_root_height"](env).tolist(), [False, False])

    def test_goal_requires_both_feet_and_upright_then_promotion_rejects_fall(self):
        env, flags, updates, _ = environment()
        data = LOGIC["route_data"](env)
        robot = env.scene["robot"].data
        robot.root_pos_w[:, 0] = env.scene.env_origins[:, 0] + data["goal_x_m"]
        robot.body_pos_w[:, :, 0] = robot.root_pos_w[:, None, 0]
        self.assertEqual(LOGIC["training_stair_goal"](env).tolist(), [True, True])
        env.scene["contact_forces"].data.net_forces_w[0] = 0.
        self.assertEqual(LOGIC["training_stair_goal"](env).tolist(), [False, True])
        env.scene["contact_forces"].data.net_forces_w[0, :, 2] = 50.
        robot.body_pos_w[0, 0, 0] = env.scene.env_origins[0, 0]
        self.assertEqual(LOGIC["training_stair_goal"](env).tolist(), [False, True])
        flags["stair_goal"][:] = True
        flags["root_height"][0] = True
        LOGIC["stair_progress_curriculum"](env, torch.tensor([0, 1]))
        self.assertEqual(updates[-1][0].tolist(), [False, True])
        self.assertEqual(updates[-1][1].tolist(), [True, False])

    def test_standing_never_promotes_and_initial_reset_does_not_demote(self):
        env, _, updates, _ = environment()
        env.episode_length_buf[0] = 0
        LOGIC["stair_progress_curriculum"](env, torch.tensor([0, 1]))
        self.assertEqual(updates[-1][0].tolist(), [False, False])
        self.assertEqual(updates[-1][1].tolist(), [False, True])

    def test_air_reward_only_landing_is_bounded_and_requires_actual_motion(self):
        env, _, _, command = environment()
        contact = torch.tensor([[True, False], [True, True]])
        sensor = SimpleNamespace(data=SimpleNamespace(last_air_time=torch.tensor([[100., 100.], [.4, .5]])),
                                 compute_first_contact=lambda dt: contact)
        env.scene["contact"] = sensor
        sensor_cfg = SimpleNamespace(name="contact", body_ids=[0, 1])
        reward = lambda: LOGIC["moving_feet_air_time"](env, "base_velocity", sensor_cfg)
        self.assertEqual(reward().tolist(), [0., 0.])
        env.scene["robot"].data.root_lin_vel_b[:, 0] = .5
        self.assertTrue(torch.allclose(reward(), torch.tensor([.2, .4])))
        contact[:] = False
        self.assertEqual(reward().tolist(), [0., 0.])
        contact[:] = True
        command[:] = 0.
        self.assertEqual(reward().tolist(), [0., 0.])

    def test_deficit_stationary_tracking_backwards_and_zero_command(self):
        env, _, _, command = environment(levels=(0, 0, 0, 0), columns=(0, 0, 0, 0))
        env.scene["robot"].data.root_lin_vel_b[:, 0] = torch.tensor([0., .5, -100., 0.])
        command[3] = 0.
        self.assertEqual(LOGIC["velocity_deficit"](env, "base_velocity").tolist(), [1., 0., 2., 0.])


class PolicyTests(unittest.TestCase):
    def test_bounds_and_backward_preserve_mean_policy_and_state_dict(self):
        class BasePolicy(torch.nn.Module):
            def __init__(self, init_noise_std=.35):
                super().__init__()
                self.std = torch.nn.Parameter(torch.ones(29) * init_noise_std)
                self.actor = torch.nn.Linear(3, 29)

            def update_distribution(self, obs):
                mean = self.actor(obs)
                self.distribution = torch.distributions.Normal(mean, mean * 0.0 + self.std)

        ns = {"EncoderMoEActorCritic": BasePolicy, "builtins": builtins, "math": math, "torch": torch}
        path = ROOT / "source/instinctlab/instinctlab/utils/bounded_stair_policy.py"
        load_definitions(path, {"BoundedStairActorCritic"}, ns)
        policy = ns["BoundedStairActorCritic"]()
        with torch.no_grad():
            policy.std[:3] = torch.tensor([-1., 10., .2])
        policy.clip_std(min=1e-12)
        self.assertTrue(torch.allclose(policy.std[:3], torch.tensor([.05, .5, .2])))
        obs = torch.ones(4, 3)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            policy.update_distribution(obs)
            loss = -policy.distribution.log_prob(torch.zeros(4, 29)).mean() - .001 * policy.distribution.entropy().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(policy.std.grad).all())
            optimizer.step()
            policy.clip_std(min=.05)
        policy.eval()
        policy.update_distribution(obs)
        self.assertTrue(torch.all(policy.std <= .50))
        self.assertTrue(torch.all(policy.std >= .05))
        self.assertEqual(set(policy.state_dict()), {"std", "actor.weight", "actor.bias"})
        with torch.no_grad():
            policy.std[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            policy.clip_std()


class RuntimeTests(unittest.TestCase):
    def test_actual_algorithm_implementations_are_saved_and_units_are_explicit(self):
        class PPO:
            def process_env_step(self):
                pass

            def update(self):
                pass

        class AMP(PPO):
            def compute_auxiliary_reward(self):
                pass

        algorithm = AMP()
        algorithm.discriminator_reward_coef = .005
        algorithm.actor_critic = SimpleNamespace(min_noise_std=.05, max_noise_std=.5)
        env_cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005), decimation=4, amp_reward_rate=.25)
        with tempfile.TemporaryDirectory() as directory:
            save_runtime_sources(SimpleNamespace(alg=algorithm), env_cfg, directory)
            output = Path(directory) / "params"
            self.assertTrue((output / "PPO_process_env_step.txt").is_file())
            self.assertTrue((output / "AMP_compute_auxiliary_reward.txt").is_file())
            self.assertIn('"step_dt": 0.02', (output / "stair_runtime.json").read_text())


if __name__ == "__main__":
    unittest.main()
