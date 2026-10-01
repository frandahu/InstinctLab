"""CPU checks for case geometry and terminal-state bookkeeping, not Isaac Sim integration."""

import ast
import importlib.util
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from stair_eval_cases import classify_episode, make_cases, summarize, surface_height
from eval_stairs import load_policy_weights
from mp4_video import Mp4Recorder, first_render_frame, video_camera_pose


def cases(num_envs=4, irregular=True):
    return make_cases(num_envs, 42, "mixed", 8, (0.08, 0.20), (0.25, 0.40), 2.0, irregular)


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

    def make_env(self):
        torch = self.torch
        staircase_cases = cases(2)
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
        for case in cases(2):
            eye, target = video_camera_pose(case, -12.0)
            self.assertEqual(target[1], -12.0)
            self.assertLess(eye[1], target[1])
            self.assertGreater(eye[2], max(case["start_height_m"], case["end_height_m"]))

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
