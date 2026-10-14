"""CPU contracts for visual actors using the existing foot-state route protocol."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_demo


class PolicyCompatibilityTests(unittest.TestCase):
    def test_visual_task_preserves_the_downstairs_demo_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            manifests = []
            for task in ("Instinct-Parkour-Mixed-Amp-G1-v0", "Instinct-Parkour-Visual-Foothold-Amp-G1-v0"):
                output = Path(directory) / task
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("run_demo.py")),
                     "--task", task, "--load_run", "unused", "--checkpoint", "model_30000.pt",
                     "--device", "cuda:0", "--headless", "--seed", "42",
                     "--terrain_mode", "stairs", "--stair_mode", "down", "--num_steps", "8",
                     "--num_envs", "2", "--episodes_per_env", "1",
                     "--episode_length_s", "90", "--max_steps", "4500",
                     "--video", "--video_view", "lane", "--video_env_id", "0",
                     "--output_dir", str(output), "--dry_run"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                manifest = json.loads((output / "cases.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["arguments"]["task"], task)
                manifests.append(manifest)
            generic = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] / "instinct_rl/eval_stairs.py"),
                 "--task", "Instinct-Parkour-Visual-Foothold-Amp-G1-v0", "--load_run", "unused",
                 "--checkpoint", "model_30000.pt", "--dry_run"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(generic.returncode, 2)
            self.assertIn("Unsupported evaluation task", generic.stderr)
        self.assertEqual(manifests[0]["cases"], manifests[1]["cases"])
        self.assertEqual(len(manifests[1]["cases"]), 2)
        for case in manifests[1]["cases"]:
            self.assertEqual(case["direction"], "down")
            self.assertGreater(case["start_height_m"], case["end_height_m"])

    def test_visual_actor_requires_its_ordered_input_without_critic_velocity(self):
        terms = ["base_ang_vel", "projected_gravity", "velocity_commands",
                 "joint_pos", "joint_vel", "actions", "depth_image", "visual_foothold"]
        base = SimpleNamespace(observation_manager=SimpleNamespace(active_terms={"policy": terms}))
        run_demo.evaluation.validate_policy_observations(base, additional_terms=("visual_foothold",))
        for invalid in (terms[:-1], [terms[-1], *terms[:-1]], ["base_lin_vel", *terms]):
            base.observation_manager.active_terms["policy"] = invalid
            with self.assertRaisesRegex(RuntimeError, "Unsupported policy observation"):
                run_demo.evaluation.validate_policy_observations(base, additional_terms=("visual_foothold",))


if __name__ == "__main__":
    unittest.main()
