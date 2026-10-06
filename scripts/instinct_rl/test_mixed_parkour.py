"""CPU checks for reference selection and upstream-recipe isolation, not GPU training."""

import ast
import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from check_parkour_motions import (
    AUTHOR_ROOT, ROBOT, ROOT, SUBSETS, check_resume_motion_inventory, configure_training_motions, validate_selection,
)
from stair_training_runtime import save_runtime_sources

CFG = ROOT / "source/instinctlab/instinctlab/tasks/parkour/config/g1"


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.names = [j.attrib["name"] for j in ET.parse(ROBOT).getroot().findall("joint") if j.attrib["type"] != "fixed"]
        for subset in SUBSETS:
            (self.root / subset).mkdir()
            self.clip(subset)

    def clip(self, subset, **overrides):
        fields = dict(framerate=25., joint_names=np.array(self.names), joint_pos=np.zeros((8, len(self.names))),
                      base_pos_w=np.zeros((8, 3)), base_quat_w=np.tile([1., 0., 0., 0.], (8, 1)))
        fields.update(overrides)
        np.savez(self.root / subset / "clip_retargeted.npz", **fields)

    def manifest(self, **overrides):
        data = dict(selected_files=[f"{s}/clip_retargeted.npz" for s in SUBSETS])
        data.update(overrides)
        path = self.root / "selection.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def test_default_includes_curb_slope_and_both_stair_subsets(self):
        report = validate_selection(self.root)
        self.assertEqual(report["subset_counts"], {s: 1 for s in SUBSETS})
        self.assertEqual(report["checked_files"], 4)
        self.assertEqual(report["frames"], 32)

    def test_missing_subset_is_not_silently_replaced_by_stairs(self):
        (self.root / "slope/clip_retargeted.npz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "slope"):
            validate_selection(self.root)

    def test_weights_follow_actual_loader_and_missing_files_fail(self):
        selection = self.manifest(motion_weights=[2., 1., 1., 0.])
        self.assertEqual(validate_selection(self.root, selection)["initial_subset_probability"]["curb"], .5)
        self.manifest(weights=[1.] * 4)
        with self.assertRaisesRegex(ValueError, "motion_weights, not weights"):
            validate_selection(self.root, selection)
        self.manifest(motion_weights=[1., -1., 1., 1.])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_selection(self.root, selection)
        self.manifest()
        (self.root / "curb/clip_retargeted.npz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Selected motion missing"):
            validate_selection(self.root, selection)

    def test_stairs_only_or_zero_weight_nonstair_references_are_rejected(self):
        for data in (
            dict(selected_files=[f"{s}/clip_retargeted.npz" for s in SUBSETS[2:]]),
            dict(motion_weights=[0., 0., 1., 1.]),
        ):
            with self.assertRaisesRegex(ValueError, "only active GRAIL stair"):
                validate_selection(self.root, self.manifest(**data))

    def test_bad_joint_mapping_shape_finite_values_and_quaternion_fail(self):
        for overrides, message in (
            (dict(joint_names=np.array(["wrong"] + self.names[1:])), "joint mapping"),
            (dict(base_pos_w=np.zeros((7, 3))), "root trajectory"),
            (dict(joint_pos=np.full((8, len(self.names)), np.nan)), "Non-finite"),
            (dict(base_quat_w=np.zeros((8, 4))), "Non-unit"),
        ):
            self.clip("curb", **overrides)
            with self.assertRaisesRegex(ValueError, message):
                validate_selection(self.root)

    def test_resume_checks_content_and_sampling_but_allows_data_relocation(self):
        report = validate_selection(self.root)
        params = self.root / "params"
        params.mkdir()
        saved = params / "motion_inventory.json"
        saved.write_text(json.dumps(report), encoding="utf-8")
        checkpoint = self.root / "model_1.pt"
        relocated = dict(report, root="/new/location")
        check_resume_motion_inventory(checkpoint, relocated)
        self.clip("curb", joint_pos=np.ones((8, len(self.names))))
        with self.assertRaisesRegex(ValueError, "differ"):
            check_resume_motion_inventory(checkpoint, validate_selection(self.root))
        saved.unlink()
        with self.assertRaisesRegex(ValueError, "start fresh"):
            check_resume_motion_inventory(checkpoint, report)

    def test_resume_refuses_old_inventory_without_data_hashes(self):
        report = validate_selection(self.root)
        params = self.root / "params"
        params.mkdir()
        old = copy.deepcopy(report)
        for motion in old["motions"]:
            del motion["sha256"]
        (params / "motion_inventory.json").write_text(json.dumps(old), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hashes are missing"):
            check_resume_motion_inventory(self.root / "model_1.pt", report)


class RecipeTests(unittest.TestCase):
    def test_new_environment_and_runner_inherit_upstream_recipe(self):
        tree = ast.parse((CFG / "mixed_parkour_cfg.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1MixedParkourEnvCfg")
        self.assertEqual(cls.bases[0].id, "G1ParkourEnvCfg")
        post = cls.body[0]
        self.assertEqual(len(post.body), 2)  # parent init + reference selection only
        target = post.body[1].targets[0]
        self.assertEqual((target.attr, target.value.attr, target.value.value.id), ("motion_reference", "scene", "self"))
        tree = ast.parse((CFG / "agents/mixed_parkour_rl_cfg.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertEqual(cls.bases[0].id, "G1ParkourPPORunnerCfg")
        self.assertEqual(len(cls.body), 1)  # experiment name only, no AMP/noise override

    def test_motion_selection_does_not_mutate_other_tasks(self):
        tree = ast.parse((CFG / "mixed_parkour_cfg.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        namespace = dict(copy=copy, os=SimpleNamespace(environ={}, path=__import__("os").path),
                         Path=Path, __file__=str(CFG / "mixed_parkour_cfg.py"))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "mixed_reference", "exec"), namespace)
        original = SimpleNamespace(motion_buffers={"run_walk": SimpleNamespace(file_path_patterns=["stair_p1/*"])})
        result = namespace["mixed_motion_reference"](original)
        self.assertEqual(original.motion_buffers["run_walk"].file_path_patterns, ["stair_p1/*"])
        motion = result.motion_buffers["run_walk"]
        self.assertEqual(Path(motion.path), AUTHOR_ROOT)
        self.assertEqual(Path(motion.filtered_motion_selection_filepath), AUTHOR_ROOT / "parkour_motion_without_run.yaml")
        self.assertIsNone(motion.file_path_patterns)

    def test_explicit_root_override_does_not_load_default_author_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = SimpleNamespace(path=str(AUTHOR_ROOT), filtered_motion_selection_filepath=str(AUTHOR_ROOT / "parkour_motion_without_run.yaml"))
            env = SimpleNamespace(scene=SimpleNamespace(
                motion_reference=SimpleNamespace(motion_buffers={"run_walk": motion}),
                robot=SimpleNamespace(spawn=SimpleNamespace(asset_path=str(ROBOT)))))
            with self.assertRaisesRegex(FileNotFoundError, "curb"):
                configure_training_motions(env, motion_root=str(root))
            self.assertIsNone(motion.filtered_motion_selection_filepath)

    def test_runtime_capture_accepts_original_policy_without_noise_bounds(self):
        algo = SimpleNamespace(actor_critic=SimpleNamespace(), discriminator_reward_coef=.25)
        env = SimpleNamespace(sim=SimpleNamespace(dt=.005), decimation=4)
        with tempfile.TemporaryDirectory() as directory:
            save_runtime_sources(SimpleNamespace(alg=algo), env, directory, filename="parkour_runtime.json")
            report = json.loads((Path(directory) / "params/parkour_runtime.json").read_text())
            self.assertEqual(report["discriminator_reward_coef"], .25)
            self.assertIsNone(report["min_noise_std"])
            self.assertIsNone(report["amp_reward_rate"])


if __name__ == "__main__":
    unittest.main()
