"""CPU checks for case geometry and terminal-state bookkeeping, not Isaac Sim integration."""

import ast
import contextlib
import importlib.util
import io
import json
import math
import random
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stair_eval_cases import classify_episode, make_cases, make_flat_control_cases, summarize, surface_height
from policy_diagnostics import policy_step_diagnostics
from eval_stairs import build_inference_policy, build_parser, load_policy_weights, validate_policy_observations
from mp4_video import Mp4Recorder, first_render_frame, video_camera_pose
from eval_startup import StartupDiagnostics
from training_yaml import load_training_yaml, restore_training_env_config
import eval_stairs
import render_bootstrap
from inspect_parkour_training import read_training_metrics, saved_reward_settings
import inspect_parkour_training


def cases(num_envs=4, irregular=True, mode="mixed"):
    return make_cases(num_envs, 42, mode, 8, (0.08, 0.20), (0.25, 0.40), 2.0, irregular)


class StartupTests(unittest.TestCase):
    def test_repeated_dry_run_preserves_old_results_and_allocates_new_directories(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            requested = Path(directory) / "up_down_video"
            args = SimpleNamespace(output_dir=requested, dry_run=True, seed=42)
            with mock.patch.object(eval_stairs, "parse_args", return_value=(args, cases(1, mode="up_down"), None)):
                eval_stairs.main()
                original = (requested / "cases.json").read_bytes()
                (requested / "staircase.mp4").write_bytes(b"existing video")
                eval_stairs.main()
                eval_stairs.main()
            self.assertEqual((requested / "cases.json").read_bytes(), original)
            self.assertEqual((requested / "staircase.mp4").read_bytes(), b"existing video")
            for suffix in ("_001", "_002"):
                actual = requested.with_name(requested.name + suffix)
                manifest = json.loads((actual / "cases.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["output_dir"], str(actual.resolve()))
                self.assertEqual(manifest["cases"][0]["direction"], "up_down")

    def test_failed_phase_and_traceback_survive_context_exit(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "saved configuration"):
                with StartupDiagnostics(output) as diagnostics:
                    diagnostics.phase("apply_saved_env", "env.yaml")
                    raise ValueError("saved configuration")
            status = json.loads((output / "startup_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "apply_saved_env")
            self.assertEqual(status["status"], "error")
            self.assertIn("ValueError: saved configuration", (output / "startup_error.txt").read_text(encoding="utf-8"))

    def test_main_reports_failure_before_app_shutdown(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            output = Path(directory) / "new_run"
            args = SimpleNamespace(output_dir=output, dry_run=False, seed=42)

            def evaluation(args, cases, output, diagnostics):
                diagnostics.phase("load_saved_env")
                raise RuntimeError("configuration loading failed")

            def close():
                # Isaac shutdown must already have a durable traceback to report,
                # even if shutdown exits before Python can print the exception.
                self.assertIn("configuration loading failed", (output / "startup_error.txt").read_text(encoding="utf-8"))
                raise SystemExit(0)

            app = SimpleNamespace(close=close)
            with mock.patch.object(eval_stairs, "parse_args", return_value=(args, [], lambda args: SimpleNamespace(app=app))), mock.patch.object(eval_stairs, "run_evaluation", side_effect=evaluation):
                with self.assertRaises(SystemExit):
                    eval_stairs.main()
            status = json.loads((output / "startup_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["error"], "RuntimeError: configuration loading failed")
            self.assertEqual(status["phase"], "load_saved_env")

    def test_successful_return_does_not_claim_completed_trials(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            with StartupDiagnostics(output) as diagnostics:
                diagnostics.phase("evaluate_policy")
                diagnostics.finish()
            status = json.loads((output / "startup_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "returned")
            self.assertIsNone(status["error"])


class TrainingYamlTests(unittest.TestCase):
    def test_nested_optional_terrain_and_sensor_values_are_initialized_together(self):
        rough = SimpleNamespace(slope_threshold=None, horizontal_scale=None, vertical_scale=None)
        noise = SimpleNamespace(min_value=None, max_value=None)
        camera = SimpleNamespace(resolution=(64, 36), channels=None, noise_pipeline=[noise])
        env_cfg = SimpleNamespace(
            seed=None, scene=SimpleNamespace(
                terrain=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains={"perlin_rough": rough})),
                camera=camera,
            ),
        )
        saved = {
            "seed": 1, "scene": {
                "terrain": {"terrain_generator": {"sub_terrains": {"perlin_rough": {
                    "slope_threshold": 0.75, "horizontal_scale": 0.05, "vertical_scale": 0.005,
                }}}},
                "camera": {"resolution": (64, 36), "channels": [0, 1], "noise_pipeline": [
                    {"min_value": 0.1, "max_value": 3.0}
                ]},
            },
        }

        def strict_update(data):
            self.assertIs(data, saved)
            self.assertEqual(env_cfg.seed, 1)
            self.assertEqual(rough.slope_threshold, 0.75)
            self.assertEqual(rough.horizontal_scale, 0.05)
            self.assertEqual(rough.vertical_scale, 0.005)
            self.assertEqual((noise.min_value, noise.max_value), (0.1, 3.0))
            self.assertEqual(camera.resolution, (64, 36))
            self.assertIs(camera.noise_pipeline[0], noise)

        env_cfg.from_dict = mock.Mock(side_effect=strict_update)
        report = restore_training_env_config(env_cfg, saved)
        self.assertEqual(len(report["initialized_optional_fields"]), 7)
        self.assertIn("/scene/terrain/terrain_generator/sub_terrains/perlin_rough/slope_threshold", report["initialized_optional_fields"])
        self.assertEqual(report["skipped_saved_fields"], [])
        camera.channels.append(2)
        self.assertEqual(saved["scene"]["camera"]["channels"], [0, 1])

    def test_evaluation_skips_only_superseded_terrain_geometry(self):
        generator = SimpleNamespace(sub_terrains={"perlin_rough": SimpleNamespace(slope_threshold=None)})
        env_cfg = SimpleNamespace(
            seed=None, scene=SimpleNamespace(
                terrain=SimpleNamespace(terrain_generator=generator, friction=0.8),
                camera=SimpleNamespace(resolution=(64, 36)),
            ), actions=SimpleNamespace(scale=0.25),
        )
        saved = {
            "seed": 1, "scene": {
                "terrain": {"terrain_generator": {"obsolete_training_field": 1}, "friction": 0.9},
                "camera": {"resolution": (64, 36)},
            }, "actions": {"scale": 0.5},
        }
        env_cfg.from_dict = mock.Mock()
        report = restore_training_env_config(env_cfg, saved, skip_paths=("/scene/terrain/terrain_generator",))
        restored = env_cfg.from_dict.call_args[0][0]
        self.assertNotIn("terrain_generator", restored["scene"]["terrain"])
        self.assertEqual(restored["scene"]["terrain"]["friction"], 0.9)
        self.assertEqual(restored["scene"]["camera"], saved["scene"]["camera"])
        self.assertEqual(restored["actions"], saved["actions"])
        self.assertIn("terrain_generator", saved["scene"]["terrain"])
        self.assertIs(env_cfg.scene.terrain.terrain_generator, generator)
        self.assertEqual(report["skipped_saved_fields"], ["/scene/terrain/terrain_generator"])

    def test_optional_priming_does_not_disable_schema_or_concrete_type_checks(self):
        env_cfg = SimpleNamespace(seed=None, camera=None, action_scale=0.25)
        saved = {"seed": 1, "camera": {"resolution": (64, 36)}, "action_scale": "invalid", "unknown": 5}

        def strict_update(data):
            self.assertIsNone(env_cfg.camera)  # Never replace a typed config with a raw dict.
            self.assertEqual(env_cfg.action_scale, 0.25)
            self.assertFalse(hasattr(env_cfg, "unknown"))
            raise ValueError("concrete field type mismatch")

        env_cfg.from_dict = mock.Mock(side_effect=strict_update)
        with self.assertRaisesRegex(ValueError, "concrete field type"):
            restore_training_env_config(env_cfg, saved)

    def test_saved_integer_seed_is_initialized_before_strict_config_restore(self):
        saved = {"seed": 1, "scene": {"camera": {"resolution": (64, 36)}}, "action_scale": 0.25}
        env_cfg = SimpleNamespace(seed=None)

        def restore(data):
            self.assertIs(data, saved)
            self.assertEqual(env_cfg.seed, 1)
            self.assertIs(type(env_cfg.seed), int)

        env_cfg.from_dict = mock.Mock(side_effect=restore)
        restore_training_env_config(env_cfg, saved)
        env_cfg.from_dict.assert_called_once_with(saved)
        # Evaluation later overrides the restored seed with --seed, as before.
        env_cfg.seed = 42
        self.assertEqual(saved["seed"], 1)

    def test_seed_defaults_invalid_values_and_other_config_errors(self):
        env_cfg = SimpleNamespace(seed=None, from_dict=mock.Mock())
        restore_training_env_config(env_cfg, {"seed": None})
        self.assertIsNone(env_cfg.seed)
        restore_training_env_config(env_cfg, {"scene": {}})
        self.assertIsNone(env_cfg.seed)
        for seed in ("42", 1.5, True):
            with self.assertRaises(ValueError):
                restore_training_env_config(env_cfg, {"seed": seed})
        env_cfg.from_dict = mock.Mock(side_effect=ValueError("camera dimensions do not match"))
        with self.assertRaisesRegex(ValueError, "camera dimensions"):
            restore_training_env_config(env_cfg, {"seed": 1, "scene": {"camera": {}}})

    def test_saved_scene_entity_slice_roundtrip_preserves_dimensions(self):
        import yaml

        original = {
            "observations": {"policy": {"joint_pos": {"params": {
                "asset_cfg": {"joint_ids": slice(None), "body_ids": slice(1, 14, 2), "name": "robot"}
            }}}},
            "camera": {"resolution": (64, 36), "clip_range": (0.1, 3.0)},
            "action_scale": 0.25,
        }
        constructors_before = dict(yaml.FullLoader.yaml_constructors)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.yaml"
            path.write_text(yaml.dump(original), encoding="utf-8")
            restored = load_training_yaml(path)
        self.assertEqual(restored, original)
        ids = restored["observations"]["policy"]["joint_pos"]["params"]["asset_cfg"]["body_ids"]
        self.assertEqual(list(range(29))[ids], list(range(1, 14, 2)))
        self.assertEqual(yaml.FullLoader.yaml_constructors, constructors_before)

    def test_slice_aliases_and_regular_agent_yaml(self):
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.yaml"
            path.write_text(
                "joint_ids: &all !!python/object/apply:builtins.slice [null, null, null]\n"
                "body_ids: *all\n", encoding="utf-8"
            )
            restored = load_training_yaml(path)
            self.assertEqual(restored["joint_ids"], slice(None))
            self.assertIs(restored["joint_ids"], restored["body_ids"])
            agent = {"device": "cuda:1", "policy": {"hidden_dims": [512, 256, 128]}, "normalize": True}
            path.write_text(yaml.safe_dump(agent), encoding="utf-8")
            self.assertEqual(load_training_yaml(path), agent)

    def test_invalid_slices_and_unrelated_object_constructors_are_rejected(self):
        import yaml

        for text in (
            "ids: !!python/object/apply:builtins.slice []\n",
            "ids: !!python/object/apply:builtins.slice [1, 2, 0]\n",
            "ids: !!python/object/apply:builtins.slice ['invalid', null, null]\n",
            "ids: !!python/object/apply:builtins.list [[1, 2]]\n",
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "env.yaml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(yaml.constructor.ConstructorError):
                    load_training_yaml(path)


class CaseTests(unittest.TestCase):
    def test_flat_control_preserves_route_seed_and_has_no_elevation(self):
        stairs = cases(2, mode="up_down")
        original = json.dumps(stairs)
        flat = make_flat_control_cases(stairs)
        self.assertEqual(json.dumps(stairs), original)
        for source, control in zip(stairs, flat):
            for key in ("case_seed", "goal_x_m", "width_m", "lane_max_x_m", "tread_depths_m"):
                self.assertEqual(control[key], source[key])
            self.assertEqual(control["direction"], "flat")
            self.assertFalse(control["requires_summit"])
            self.assertEqual(control["flights"], [])
            for x in (-1.0, 0.0, 1.21, 3.0, source["goal_x_m"]):
                self.assertEqual(surface_height(control, x), 0.0)

    def test_seed_and_lane_extension(self):
        self.assertEqual(cases(), cases())
        self.assertEqual(cases(), cases(8)[:4])
        self.assertNotEqual(cases()[0]["riser_heights_m"], cases()[2]["riser_heights_m"])

    def test_up_and_down_geometry(self):
        for case in cases():
            edge = case["stair_start_x_m"]
            last_height = case["start_height_m"]
            sign = 1 if case["direction"] == "up" else -1
            self.assertAlmostEqual(surface_height(case, 0.0), last_height)
            for rise, depth, level in zip(
                case["riser_heights_m"], case["tread_depths_m"], case["surface_heights_m"]
            ):
                self.assertAlmostEqual(level - last_height, sign * rise)
                self.assertAlmostEqual(surface_height(case, edge + depth / 2), level)
                last_height = level
                edge += depth
            self.assertAlmostEqual(edge, case["stair_end_x_m"])
            self.assertAlmostEqual(surface_height(case, case["goal_x_m"]), case["end_height_m"])
            if sign == -1:
                self.assertAlmostEqual(case["end_height_m"], 0.0)

    def test_regular_flight_and_invalid_ranges(self):
        self.assertEqual(len(set(cases(irregular=False)[0]["riser_heights_m"])), 1)
        for heights in ((0, 0.2), (0.2, 0.1), (0.1, float("nan"))):
            with self.assertRaises(ValueError):
                make_cases(2, 42, "mixed", 8, heights, (0.25, 0.4), 2.0, False)

    def test_up_down_route_and_sparse_anomalies(self):
        for seed in range(30):
            for case in make_cases(3, seed, "up_down", 8, (0.08, 0.20), (0.25, 0.40), 2.0, True):
                self.assertEqual(case["start_height_m"], 0)
                self.assertEqual(case["end_height_m"], 0)
                self.assertTrue(case["requires_summit"])
                self.assertEqual(len(case["riser_heights_m"]), 17)
                for flight in case["flights"]:
                    changed = []
                    for index, (height, depth) in enumerate(zip(flight["riser_heights_m"], flight["tread_depths_m"])):
                        self.assertTrue(0.08 <= height <= 0.20)
                        self.assertTrue(0.25 <= depth <= 0.40)
                        if height != case["nominal_riser_height_m"] or depth != case["nominal_tread_depth_m"]:
                            changed.append(index)
                    self.assertEqual(changed, flight["anomalous_step_indices"])
                    self.assertEqual(len(changed), 2)
                up, down = case["flights"]
                self.assertAlmostEqual(sum(up["riser_heights_m"]), sum(down["riser_heights_m"]))
                self.assertEqual(case["riser_deltas_m"][8], 0)
                self.assertTrue(all(delta > 0 for delta in case["riser_deltas_m"][:8]))
                self.assertTrue(all(delta < 0 for delta in case["riser_deltas_m"][9:]))
                edge, last = case["stair_start_x_m"], 0.0
                for delta, depth, level in zip(case["riser_deltas_m"], case["tread_depths_m"], case["surface_heights_m"]):
                    self.assertAlmostEqual(level - last, delta)
                    self.assertAlmostEqual(surface_height(case, edge + depth / 2), level)
                    edge, last = edge + depth, level
                self.assertAlmostEqual(edge, case["stair_end_x_m"])
                center = (case["summit_start_x_m"] + case["summit_end_x_m"]) / 2
                self.assertAlmostEqual(surface_height(case, center), case["summit_height_m"])

    def test_sparse_boundaries_fixed_dimensions_and_defaults(self):
        args = build_parser().parse_args(["--load_run", "unused", "--checkpoint", "model.pt"])
        self.assertEqual(args.stair_mode, "up_down")
        self.assertEqual(args.terrain_mode, "stairs")
        self.assertFalse(args.sample)
        self.assertTrue(args.irregular)
        self.assertEqual(args.episode_length_s, 45)
        for height_range, depth_range in (((0.12, 0.12), (0.25, 0.4)), ((0.08, 0.2), (0.3, 0.3))):
            for count in (3, 4, 8):
                for case in make_cases(20, 42, "up_down", count, height_range, depth_range, 2, True, 0.49):
                    self.assertLess(case["anomalous_steps_per_flight"], count / 2)
                    for flight in case["flights"]:
                        changed = sum(h != case["nominal_riser_height_m"] or d != case["nominal_tread_depth_m"]
                                      for h, d in zip(flight["riser_heights_m"], flight["tread_depths_m"]))
                        self.assertEqual(changed, case["anomalous_steps_per_flight"])
        with self.assertRaises(ValueError):
            make_cases(1, 42, "up_down", 2, (0.08, 0.2), (0.25, 0.4), 2, True)
        with self.assertRaises(ValueError):
            make_cases(1, 42, "up_down", 8, (0.12, 0.12), (0.3, 0.3), 2, True)
        regular = cases(1, irregular=False, mode="up_down")[0]
        self.assertTrue(all(not flight["anomalous_step_indices"] for flight in regular["flights"]))

    def test_failure_precedes_success_and_timeout(self):
        self.assertEqual(classify_episode({"stair_success": True, "base_contact": True}), "base_contact")
        self.assertEqual(classify_episode({"stair_success": True, "time_out": True}), "success")
        self.assertEqual(classify_episode({"time_out": True}), "timeout")

    def test_incomplete_trials_have_separate_denominators(self):
        report = summarize([{"outcome": "success", "elapsed_s": 10}], 4)
        self.assertEqual(report["success_rate_completed"], 1.0)
        self.assertEqual(report["success_rate_planned"], 0.25)
        self.assertEqual(report["uncompleted_episodes"], 3)
        self.assertIsNone(summarize([], 4)["success_rate_completed"])


class TensorMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        path = Path(__file__).resolve().parents[2] / "source/instinctlab/instinctlab/tasks/parkour/config/g1/stair_eval_cfg.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"_case_tensors", "ground_height", "reset_stair_root", "stair_root_height", "StairSuccess", "straight_velocity_command"}
        # Exercise tensor math without importing the absent Isaac Sim runtime.
        metric_tree = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])

        class TermBase:
            def __init__(self, cfg, env):
                pass

        def yaw_quat(roll, pitch, yaw):
            return torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)

        cls.namespace = {
            "torch": torch, "math": math, "random": random, "ManagerTermBase": TermBase,
            "quat_from_euler_xyz": yaw_quat,
            # Reset tests use an identity default orientation, so this is q * I.
            "quat_mul": lambda q, identity: q,
        }
        exec(compile(metric_tree, str(path), "exec"), cls.namespace)

    def make_env(self, mode="mixed"):
        torch = self.torch
        staircase_cases = cases(2, mode=mode)
        origins = torch.tensor([[0, 0, case["start_height_m"]] for case in staircase_cases])
        data = SimpleNamespace(
            root_pos_w=torch.tensor([[case["goal_x_m"], 0, case["end_height_m"] + 0.9] for case in staircase_cases]),
            body_pos_w=torch.tensor([
                [[case["goal_x_m"] - 0.1, -0.1, case["end_height_m"] + 0.06],
                 [case["goal_x_m"] + 0.1, 0.1, case["end_height_m"] + 0.06]] for case in staircase_cases
            ]),
            projected_gravity_b=torch.tensor([[0, 0, -1.0], [0, 0, -1.0]]),
            root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
            root_lin_vel_w=torch.zeros(2, 3), heading_w=torch.zeros(2),
            joint_vel=torch.zeros(2, 29), applied_torque=torch.zeros(2, 29),
        )
        robot = SimpleNamespace(data=data, find_bodies=lambda *args, **kwargs: ([0, 1], []))
        contact = SimpleNamespace(
            data=SimpleNamespace(net_forces_w=torch.tensor([[[0, 0, 50.0], [0, 0, 50.0]]] * 2)),
            find_bodies=lambda *args, **kwargs: ([0, 1], []),
        )

        class Scene(dict):
            pass

        scene = Scene(robot=robot, contact_forces=contact)
        scene.env_origins = origins
        scene.terrain = SimpleNamespace(
            terrain_generator=SimpleNamespace(cases=staircase_cases), terrain_types=torch.tensor([0, 1])
        )
        return SimpleNamespace(
            scene=scene, num_envs=2, device="cpu", step_dt=0.02, episode_length_buf=torch.ones(2, dtype=torch.long),
            command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(2, 3)),
            _stair_spawn_offsets=torch.zeros(2, 3),
        )

    def test_tensor_ground_and_high_platform_fall(self):
        env = self.make_env()
        torch = self.torch
        for offset in (-0.5, 1.21, 1.7, 3.0, 5.5):
            measured = self.namespace["ground_height"](env, torch.tensor([offset, offset])).tolist()
            for case, height in zip(cases(2), measured):
                self.assertAlmostEqual(height, surface_height(case, offset), places=5)
        # Fall on the elevated landing: absolute z is high, clearance is low.
        env.scene["robot"].data.root_pos_w[0, 2] = cases(2)[0]["end_height_m"] + 0.3
        self.assertEqual(self.namespace["stair_root_height"](env).tolist(), [True, False])

    def test_dwell_and_terminal_snapshot_survive_reset(self):
        env = self.make_env()
        env.scene["robot"].data.joint_vel[:] = 2.0
        env.scene["robot"].data.applied_torque[:] = 5.0
        term = self.namespace["StairSuccess"](None, env)
        for _ in range(24):
            self.assertEqual(term(env).tolist(), [False, False])
        self.assertEqual(term(env).tolist(), [True, True])
        terminal = term.snapshot["pos"].clone()
        term.reset([0, 1])
        env.scene["robot"].data.root_pos_w[:] = 0
        self.assertTrue(self.torch.equal(term.snapshot["pos"], terminal))
        env.scene["robot"].data.joint_vel[:] = 0.0
        self.assertEqual(term.snapshot["joint_velocity_rms_rad_s"].tolist(), [2.0, 2.0])
        self.assertEqual(term.snapshot["applied_torque_rms_nm"].tolist(), [5.0, 5.0])
        self.assertEqual(term.hold_steps.tolist(), [0, 0])

    def test_straight_command_wraps_heading_clips_yaw_and_never_requests_lateral_motion(self):
        torch = self.torch
        heading = torch.tensor([0., .1, -.1, math.pi / 2, 2 * math.pi + .1])
        command = self.namespace["straight_velocity_command"](heading, .5, 1., 2.)
        self.assertTrue(torch.allclose(command[:, 0], torch.full((5,), .5)))
        self.assertTrue(torch.equal(command[:, 1], torch.zeros(5)))
        self.assertTrue(torch.allclose(command[:, 2], torch.tensor([0., -.2, .2, -1., -.2]), atol=1e-5))

    def test_heading_and_world_velocity_snapshot_are_separate_from_body_velocity(self):
        env = self.make_env()
        data = env.scene["robot"].data
        data.root_lin_vel_b[:] = self.torch.tensor([.5, 0., 0.])
        data.root_lin_vel_w[:] = self.torch.tensor([0., .5, 0.])
        data.heading_w[:] = math.pi / 2
        term = self.namespace["StairSuccess"](None, env)
        term(env)
        self.assertTrue(self.torch.equal(term.snapshot["velocity"], data.root_lin_vel_b))
        self.assertTrue(self.torch.equal(term.snapshot["velocity_w"], data.root_lin_vel_w))
        data.heading_w.zero_()
        self.assertTrue(self.torch.allclose(term.snapshot["heading_w"], self.torch.full((2,), math.pi / 2)))

    def test_flat_control_success_does_not_require_stair_summit(self):
        env = self.make_env(mode="up_down")
        env.scene.terrain.terrain_generator.cases = make_flat_control_cases(cases(2, mode="up_down"))
        term = self.namespace["StairSuccess"](None, env)
        for _ in range(25):
            result = term(env)
        self.assertEqual(result.tolist(), [True, True])
        self.assertEqual(term.summit_reached.tolist(), [False, False])

    def test_policy_diagnostics_use_actual_command_history_and_depth_input(self):
        torch = self.torch
        fmt = {"base_ang_vel": (2,), "velocity_commands": (6,), "depth_image": (2, 1, 2)}
        observation = torch.tensor([[99, 98, 0, 0, 0, 0.5, 0, 0.1, 0.1, 0.2, 0.3, 0.4]])
        actions, previous = torch.tensor([[3.0, 4.0]]), torch.tensor([[2.0, 3.0]])
        detail = policy_step_diagnostics(observation, actions, previous, fmt)
        self.assertEqual(detail["observed_command_x_min_m_s"].item(), 0.0)
        self.assertEqual(detail["observed_command_x_max_m_s"].item(), 0.5)
        self.assertAlmostEqual(detail["depth_input_min"].item(), 0.1)
        self.assertAlmostEqual(detail["depth_input_max"].item(), 0.4)
        self.assertAlmostEqual(detail["depth_input_mean"].item(), 0.25)
        self.assertAlmostEqual(detail["policy_action_rms"].item(), math.sqrt(12.5), places=5)
        self.assertEqual(detail["policy_action_delta_rms"].item(), 1.0)
        initial = policy_step_diagnostics(observation, actions, None, fmt)
        self.assertEqual(initial["policy_action_delta_rms"].item(), 0.0)
        with self.assertRaisesRegex(ValueError, "width"):
            policy_step_diagnostics(observation[:, :-1], actions, previous, fmt)

    def test_complete_route_requires_supported_summit_before_final_goal(self):
        env = self.make_env(mode="up_down")
        term = self.namespace["StairSuccess"](None, env)
        torch = self.torch
        # Final-goal coordinates alone must never satisfy the combined route.
        for _ in range(30):
            self.assertEqual(term(env).tolist(), [False, False])
        self.assertEqual(term.summit_reached.tolist(), [False, False])
        final_pos = env.scene["robot"].data.root_pos_w.clone()
        final_feet = env.scene["robot"].data.body_pos_w.clone()
        for env_id, case in enumerate(cases(2, mode="up_down")):
            x = (case["summit_start_x_m"] + case["summit_end_x_m"]) / 2
            env.scene["robot"].data.root_pos_w[env_id] = torch.tensor([x, 0, case["summit_height_m"] + 0.9])
            env.scene["robot"].data.body_pos_w[env_id, :, 0] = torch.tensor([x - 0.1, x + 0.1])
            env.scene["robot"].data.body_pos_w[env_id, :, 2] = case["summit_height_m"] + 0.06
        env.scene["contact_forces"].data.net_forces_w[:] = 0
        term(env)
        self.assertEqual(term.summit_reached.tolist(), [False, False])
        env.scene["contact_forces"].data.net_forces_w[:, :, 2] = 50
        term(env)
        self.assertEqual(term.summit_reached.tolist(), [True, True])
        env.scene["robot"].data.root_pos_w[:] = final_pos
        env.scene["robot"].data.body_pos_w[:] = final_feet
        for _ in range(24):
            self.assertEqual(term(env).tolist(), [False, False])
        self.assertEqual(term(env).tolist(), [True, True])
        term.reset([0])
        self.assertEqual(term.summit_reached.tolist(), [False, True])
        self.assertEqual(term.snapshot["summit_reached"].tolist(), [True, True])
        self.assertEqual(term(env).tolist(), [False, True])

    def test_combined_route_tensor_floor_matches_geometry_at_every_segment(self):
        env = self.make_env(mode="up_down")
        torch = self.torch
        routes = cases(2, mode="up_down")
        edges = [case["stair_start_x_m"] for case in routes]
        for segment in range(17):
            positions = [x + case["tread_depths_m"][segment] / 2 for x, case in zip(edges, routes)]
            measured = self.namespace["ground_height"](env, torch.tensor(positions)).tolist()
            for case, x, floor in zip(routes, positions, measured):
                self.assertAlmostEqual(floor, surface_height(case, x), places=5)
            edges = [x + case["tread_depths_m"][segment] for x, case in zip(edges, routes)]

    def test_checkpoint_load_preserves_policy_and_normalizer(self):
        torch = self.torch
        model, normalizer = torch.nn.Linear(3, 2), torch.nn.Linear(3, 3)
        saved_model, saved_normalizer = torch.nn.Linear(3, 2), torch.nn.Linear(3, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({
                "model_state_dict": saved_model.state_dict(),
                "policy_normalizer_state_dict": saved_normalizer.state_dict(), "iter": 2000,
            }, path)
            iteration = load_policy_weights(model, {"policy": normalizer}, {}, path)
            self.assertTrue(torch.equal(model.weight, saved_model.weight))
            self.assertTrue(torch.equal(normalizer.weight, saved_normalizer.weight))
            self.assertEqual(iteration, 2000)
            torch.save({"model_state_dict": saved_model.state_dict()}, path)
            with self.assertRaises(KeyError):
                load_policy_weights(model, {"policy": normalizer}, {}, path)

    def test_actor_observation_contract_excludes_critic_velocity(self):
        terms = [
            "base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos", "joint_vel", "actions", "depth_image",
        ]
        base = SimpleNamespace(observation_manager=SimpleNamespace(active_terms={"policy": terms}))
        validate_policy_observations(base)
        base.observation_manager.active_terms["policy"] = ["base_lin_vel", *terms]
        with self.assertRaisesRegex(RuntimeError, "Unsupported policy observation"):
            validate_policy_observations(base)

    def test_inference_builds_actor_without_amp_runner_or_reference_group(self):
        torch = self.torch

        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.std = torch.nn.Parameter(torch.full((2,), 0.4))

            def act_inference(self, obs):
                return self.linear(obs)

            def act(self, obs):
                return torch.distributions.Normal(self.linear(obs), self.std).sample()

        actor, normalizer = Actor(), torch.nn.Linear(3, 3)
        with torch.no_grad():
            normalizer.weight.copy_(torch.eye(3))
            normalizer.bias.zero_()
        format_seen = []

        def make_actor(class_name, cfg, obs_format, num_actions, num_rewards):
            format_seen.append(obs_format)
            self.assertEqual((class_name, num_actions, num_rewards), ("EncoderMoEActorCritic", 2, 1))
            return Actor()

        fake_modules = SimpleNamespace(
            build_actor_critic=make_actor,
            build_normalizer=lambda **kwargs: torch.nn.Linear(3, 3),
        )
        fake_package = SimpleNamespace(modules=fake_modules)
        fake_utils = SimpleNamespace(get_subobs_size=lambda segment: 3)
        env = SimpleNamespace(
            get_obs_format=lambda: {"policy": {"depth_image": (3,)}, "critic": {"base_lin_vel": (3,)}},
            num_actions=2, num_rewards=1,
        )
        cfg = {
            "policy": {"class_name": "EncoderMoEActorCritic"},
            "normalizers": {"policy": {"class_name": "EmpiricalNormalization"}, "critic": {"class_name": "unused"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({
                "model_state_dict": actor.state_dict(),
                "policy_normalizer_state_dict": normalizer.state_dict(),
                "iter": 6000,
            }, path)
            with mock.patch.dict(sys.modules, {
                "instinct_rl": fake_package,
                "instinct_rl.utils": SimpleNamespace(),
                "instinct_rl.utils.utils": fake_utils,
            }):
                policy, iteration = build_inference_policy(env, cfg, path, "cpu")
                sampled_policy, _ = build_inference_policy(env, cfg, path, "cpu", sample=True)
            self.assertEqual(iteration, 6000)
            self.assertEqual(cfg["policy"]["class_name"], "EncoderMoEActorCritic")
            self.assertEqual(list(format_seen[0]), ["policy", "critic"])
            self.assertTrue(torch.allclose(policy(torch.ones(1, 3)), actor.act_inference(torch.ones(1, 3))))
            batch = torch.ones(4096, 3)
            mean = policy(batch)
            torch.manual_seed(11)
            sampled = sampled_policy(batch)
            self.assertFalse(torch.allclose(sampled, mean))
            self.assertTrue(torch.allclose((sampled - mean).mean(dim=0), torch.zeros(2), atol=0.03))
            self.assertTrue(torch.allclose((sampled - mean).std(dim=0), actor.std, atol=0.03))

    def test_spawn_sequence_is_independent_of_other_lanes(self):
        torch = self.torch
        env_a, env_b = self.make_env(), self.make_env()
        for env in (env_a, env_b):
            robot = env.scene["robot"]
            robot.data.default_root_state = torch.zeros(2, 13)
            robot.data.default_root_state[:, 2] = 0.9
            robot.data.default_root_state[:, 3] = 1.0
            robot.write_root_pose_to_sim = lambda *args, **kwargs: None
            robot.write_root_velocity_to_sim = lambda *args, **kwargs: None
        reset = self.namespace["reset_stair_root"]
        reset(env_a, torch.tensor([0, 1]))
        reset(env_a, torch.tensor([0, 1]))
        reset(env_b, torch.tensor([0]))
        reset(env_b, torch.tensor([0]))
        reset(env_b, torch.tensor([1]))
        reset(env_b, torch.tensor([1]))
        self.assertTrue(torch.equal(env_a._stair_spawn_offsets, env_b._stair_spawn_offsets))
        reset(env_a, torch.tensor([0, 1]), centered_start=True)
        self.assertTrue(torch.equal(env_a._stair_spawn_offsets, torch.zeros(2, 3)))


class TrainingLogTests(unittest.TestCase):
    def test_setup_reads_environment_count_and_verified_reference_from_saved_files(self):
        from prepare_parkour_references import FILES

        with tempfile.TemporaryDirectory() as directory:
            params = Path(directory) / "params"
            params.mkdir()
            (params / "env.yaml").write_text("scene: {num_envs: 32}\n", encoding="utf-8")
            (params / "agent.yaml").write_text(
                "num_steps_per_env: 24\nmax_iterations: 10000\nalgorithm: {num_mini_batches: 4, discriminator_reward_coef: 0.25}\n",
                encoding="utf-8")
            (params / "motion_inventory.json").write_text(json.dumps({
                "files": 1, "frames": 18982, "motions": [{"sha256": FILES["parkour_motion_without_run_retargetted.npz"][1]}],
            }), encoding="utf-8")
            report = inspect_parkour_training.saved_training_setup(directory)
            self.assertEqual(report["num_envs"], 32)
            self.assertEqual(report["transitions_per_iteration"], 768)
            self.assertEqual(report["transitions_per_minibatch"], 192)
            self.assertTrue(report["author_motion_hash_matches"])
            (params / "motion_inventory.json").unlink()
            self.assertFalse(inspect_parkour_training.saved_training_setup(directory)["author_motion_hash_matches"])

    def test_trace_metrics_separate_episodes_body_speed_and_world_path(self):
        import csv

        with tempfile.TemporaryDirectory() as directory:
            fields = ["env_id", "episode_index", "elapsed_s", "base_x_m", "base_y_m", "vel_x_m_s", "vel_y_m_s",
                      "command_x_m_s", "command_y_m_s", "command_yaw_rad_s", "heading_w_rad"]
            with (Path(directory) / "trace.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerow([0, 0, .02, 0, 0, 0, 0, .5, 0, 0, .2])
                writer.writerow([0, 0, 1, .5, .2, .4, .1, .5, 0, -.4, .2])
                writer.writerow([0, 0, 2, 1, .4, .4, .1, .5, 0, -.4, .2])
                writer.writerow([0, 1, .02, 0, 0, 0, 0, .5, 0, 0, ""])
                writer.writerow([0, 1, 1, .1, -.1, .1, -.1, .5, 0, 0, ""])
            first, second = inspect_parkour_training.read_evaluation_metrics(directory)
            self.assertEqual(first["moving_samples"], 2)
            self.assertAlmostEqual(first["mean_forward_speed_body_m_s"], .4)
            self.assertAlmostEqual(first["mean_velocity_error_body_m_s"], math.sqrt(.02))
            self.assertAlmostEqual(first["mean_abs_command_yaw_rad_s"], .4)
            self.assertAlmostEqual(first["net_path_angle_deg"], math.degrees(math.atan2(.4, 1)))
            self.assertAlmostEqual(first["mean_abs_heading_world_deg"], math.degrees(.2))
            self.assertIsNone(second["mean_abs_heading_world_deg"])
            self.assertEqual(second["delta_x_m"], .1)

    def test_real_tensorboard_logs_filter_checkpoint_iteration_and_restart_duplicates(self):
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as directory:
            tag = "Episode_Reward/rewards_track_lin_vel_xy_exp/timestep"
            with SummaryWriter(directory) as writer:
                writer.add_scalar(tag, 0.1, 5000, walltime=1)
                writer.add_scalar(tag, 0.2, 5950, walltime=2)
                writer.add_scalar(tag, 0.3, 6000, walltime=3)
                writer.add_scalar(tag, 0.9, 6050, walltime=4)
                writer.add_scalar(tag, 0.4, 6000, walltime=5)
                writer.add_scalar("Policy/mean_noise_std", 0.5, 6000)
                writer.add_scalar("Train/mean_episode_length", 998, 6000)
                writer.add_scalar("Train/time/mean_episode_length", 777, 5980)
                writer.add_scalar("Loss/value_function", 123, 6000)
            metrics, tags = read_training_metrics(directory, samples=2, through_iteration=6000)
            self.assertEqual([step for step, _ in metrics[tag]], [5950, 6000])
            self.assertAlmostEqual(metrics[tag][-1][1], 0.4)
            self.assertEqual(metrics["Train/mean_episode_length"], [(6000, 998.0)])
            self.assertNotIn("Loss/value_function", metrics)
            self.assertNotIn("Train/time/mean_episode_length", metrics)
            self.assertIn("Loss/value_function", tags)

    def test_cpu_inspector_reads_saved_weights_and_config_without_simulator(self):
        import torch
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "params").mkdir()
            (run_dir / "params/env.yaml").write_text(
                "rewards:\n  rewards:\n    track_lin_vel_xy_exp:\n      weight: 2.0\n"
                "      params: {std: 0.5}\n    is_alive: {weight: 3.0}\n", encoding="utf-8",
            )
            path = run_dir / "model_6000.pt"
            torch.save({"iter": 6000, "model_state_dict": {"std": torch.tensor([0.2, 0.4])}}, path)
            with SummaryWriter(directory) as writer:
                writer.add_scalar("Policy/mean_noise_std", 0.3, 6000)
                writer.add_scalar("Policy/mean_noise_std", 0.1, 7000)
            before = path.read_bytes()
            with mock.patch.object(sys, "argv", [
                "inspect_parkour_training.py", "--load_run", directory, "--checkpoint", path.name,
            ]), contextlib.redirect_stdout(io.StringIO()) as output:
                inspect_parkour_training.main()
            text = output.getvalue()
            self.assertIn("[ACTION_STD] mean=0.3000", text)
            self.assertIn("[METRIC] Policy/mean_noise_std: iter=6000", text)
            self.assertNotIn("iter=7000", text)
            self.assertEqual(path.read_bytes(), before)
            settings = saved_reward_settings(directory)
            self.assertEqual(settings["rewards/track_lin_vel_xy_exp"], {"weight": 2.0, "std": 0.5})
            # A fresh standalone reader must not initialize TensorFlow or CUDA,
            # even on machines where TensorFlow is installed.
            script_dir = str(Path(__file__).resolve().parent)
            result = subprocess.run([
                sys.executable, "-c",
                "import sys; sys.path.insert(0, sys.argv.pop(1)); "
                "import inspect_parkour_training as inspector; inspector.main(); "
                "import torch; assert 'tensorflow' not in sys.modules; assert not torch.cuda.is_initialized()",
                script_dir, "--load_run", directory, "--checkpoint", path.name,
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[ACTION_STD] mean=0.3000", result.stdout)


class RenderBootstrapTests(unittest.TestCase):
    def test_numpy2_selects_only_bundled_numpy_before_simulator_import(self):
        with tempfile.TemporaryDirectory() as directory:
            default = Path(directory) / "numpy2"
            default.mkdir()
            (default / "version.py").write_text("version = '2.2.6'\n", encoding="utf-8")
            bundled = Path(directory) / "prebundle" / "numpy"
            module = SimpleNamespace(__version__="1.26.0", __file__=str(bundled / "__init__.py"))
            original_path = sys.path.copy()
            with mock.patch.dict(sys.modules), contextlib.redirect_stdout(io.StringIO()):
                sys.modules.pop("numpy", None)
                with mock.patch.object(render_bootstrap.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(default / "__init__.py"))), \
                     mock.patch.object(render_bootstrap, "_bundled_numpy_candidates", return_value=iter([bundled])), \
                     mock.patch.object(render_bootstrap, "_compatible_package", return_value=True), \
                     mock.patch.object(render_bootstrap, "_cached_numpy_package") as cache, \
                     mock.patch.object(render_bootstrap, "_load_numpy_package", return_value=module) as load:
                    record = render_bootstrap.preload_isaac_numpy()
            self.assertEqual(record["source"], "isaac_sim_bundle")
            self.assertEqual(record["version"], "1.26.0")
            load.assert_called_once_with(bundled)
            cache.assert_not_called()
            self.assertEqual(sys.path, original_path)

    def test_loaded_numpy2_is_not_unsafely_swapped(self):
        with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace(__version__="2.2.6")}):
            with self.assertRaisesRegex(RuntimeError, "fresh Python process"):
                render_bootstrap.preload_isaac_numpy()

    def test_missing_bundle_caches_one_wheel_without_global_pip_install(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_script = Path(directory) / "scripts" / "instinct_rl" / "render_bootstrap.py"
            commands = []

            def install(command, check):
                commands.append(command)
                package = Path(command[command.index("--target") + 1]) / "numpy"
                (package / "core").mkdir(parents=True)
                (package / "version.py").write_text("version = '1.26.4'\n", encoding="utf-8")
                (package / "core" / ("_multiarray_umath" + render_bootstrap.EXTENSION_SUFFIXES[0])).touch()

            with mock.patch.object(render_bootstrap, "__file__", str(fake_script)), \
                 mock.patch.object(render_bootstrap.subprocess, "run", side_effect=install), \
                 contextlib.redirect_stdout(io.StringIO()):
                package = render_bootstrap._cached_numpy_package()
                self.assertEqual(package, render_bootstrap._cached_numpy_package())
            self.assertEqual(len(commands), 1)
            self.assertIn("--no-deps", commands[0])
            self.assertEqual(commands[0][-1], "numpy==1.26.4")
            self.assertTrue(str(package).startswith(str(Path(directory) / "outputs")))

    def test_native_numpy_can_be_preloaded_without_other_package_overrides(self):
        import numpy as np

        script = (
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); from render_bootstrap import _load_numpy_package; "
            "before=sys.path.copy(); np=_load_numpy_package(Path(sys.argv[2])); "
            "assert np.array([1],dtype=np.uint64).dtype.itemsize == 8; "
            "assert np.arange(4).sum() == 6; assert sys.path == before"
        )
        subprocess.run([
            sys.executable, "-c", script, str(Path(__file__).parent), str(Path(np.__file__).parent),
        ], check=True)


class VideoTests(unittest.TestCase):
    def test_camera_frames_selected_lane(self):
        for case in cases(2) + cases(2, mode="up_down"):
            eye, target = video_camera_pose(case, -12.0)
            self.assertEqual(target[1], -12.0)
            self.assertLess(eye[1], target[1])
            self.assertGreater(eye[2], max(case["surface_heights_m"]))

    def test_renderer_warmup_and_empty_render_failure(self):
        import numpy as np

        blank = np.zeros((72, 128, 3), dtype=np.uint8)
        rgb = blank.copy()
        rgb[:, :, 0] = 200
        frames = iter([blank, blank, rgb])
        self.assertTrue(np.array_equal(first_render_frame(SimpleNamespace(render=lambda: next(frames))), rgb))
        with self.assertRaises(RuntimeError):
            first_render_frame(SimpleNamespace(render=lambda: blank))

    @unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg"), "FFmpeg package is needed for encoding integration")
    def test_real_mp4_roundtrip_and_frame_validation(self):
        import imageio
        import imageio_ffmpeg
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "staircase.mp4"
            recorder = Mp4Recorder(path, fps=25)
            with self.assertRaises(ValueError):
                recorder.append(np.zeros((71, 128, 3), dtype=np.uint8))
            for frame_id in range(6):
                frame = np.zeros((72, 128, 3), dtype=np.uint8)
                frame[:, :, 0] = 30 + frame_id * 30
                recorder.append(frame)
            recorder.close()
            recorder.close()  # Finalization is safe when cleanup runs twice.
            self.assertGreater(path.stat().st_size, 0)
            frame_count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
            self.assertEqual(frame_count, 6)
            self.assertAlmostEqual(duration, 6 / 25, places=2)
            reader = imageio.get_reader(str(path), format="FFMPEG")
            try:
                self.assertEqual(reader.get_data(0).shape, (72, 128, 3))
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
