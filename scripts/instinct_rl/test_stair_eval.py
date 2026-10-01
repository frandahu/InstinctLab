"""CPU checks for case geometry and terminal-state bookkeeping, not Isaac Sim integration."""

import ast
import contextlib
import importlib.util
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stair_eval_cases import classify_episode, make_cases, summarize, surface_height
from eval_stairs import build_parser, load_policy_weights
from mp4_video import Mp4Recorder, first_render_frame, video_camera_pose
from eval_startup import StartupDiagnostics
from training_yaml import load_training_yaml
import eval_stairs


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
        names = {"_case_tensors", "ground_height", "reset_stair_root", "stair_root_height", "StairSuccess"}
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
        term = self.namespace["StairSuccess"](None, env)
        for _ in range(24):
            self.assertEqual(term(env).tolist(), [False, False])
        self.assertEqual(term(env).tolist(), [True, True])
        terminal = term.snapshot["pos"].clone()
        term.reset([0, 1])
        env.scene["robot"].data.root_pos_w[:] = 0
        self.assertTrue(self.torch.equal(term.snapshot["pos"], terminal))
        self.assertEqual(term.hold_steps.tolist(), [0, 0])

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
        runner = SimpleNamespace(
            cfg={}, alg=SimpleNamespace(actor_critic=model), normalizers={"policy": normalizer}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({
                "model_state_dict": saved_model.state_dict(),
                "policy_normalizer_state_dict": saved_normalizer.state_dict(), "iter": 2000,
            }, path)
            load_policy_weights(runner, path)
            self.assertTrue(torch.equal(model.weight, saved_model.weight))
            self.assertTrue(torch.equal(normalizer.weight, saved_normalizer.weight))
            self.assertEqual(runner.current_learning_iteration, 2000)
            torch.save({"model_state_dict": saved_model.state_dict()}, path)
            with self.assertRaises(KeyError):
                load_policy_weights(runner, path)

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
