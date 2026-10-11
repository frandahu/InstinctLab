"""CPU contracts for retaining both AMP expert sources and their weights."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "instinct_rl"))
from check_parkour_motions import AUTHOR_SELECTION, ROBOT, check_resume_motion_inventory, validate_selection
from prepare_visual_motions import prepare_visual_motions


class JointMotionPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.author = self.root / "author"
        self.grail = self.root / "grail"
        self.output = self.root / "manifests"
        names = [j.attrib["name"] for j in ET.parse(ROBOT).getroot().findall("joint")
                 if j.attrib.get("type") != "fixed"]
        self.clip = dict(framerate=np.array(50.0), joint_names=np.array(names),
                         joint_pos=np.zeros((3, len(names))), base_pos_w=np.zeros((3, 3)),
                         base_quat_w=np.tile([1., 0., 0., 0.], (3, 1)))
        self.author_files = [self.write_clip(self.author / f"walk_{i}_retargeted.npz") for i in range(2)]
        (self.author / AUTHOR_SELECTION).write_text(yaml.safe_dump({
            "selected_files": [p.name for p in self.author_files], "motion_weights": [3., 1.],
        }), encoding="utf-8")
        self.grail_files = [self.write_clip(self.grail / name / "step_retargeted.npz")
                            for name in ("curb", "slope", "stair_p1", "stair_p2")]
        self.grail_files.append(self.write_clip(self.grail / "curb/extra_retargeted.npz"))

    def write_clip(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.clip)
        return path.resolve()

    def prepare(self, **kwargs):
        return prepare_visual_motions(self.grail, author_root=self.author, output_dir=self.output, **kwargs)

    def test_both_sources_reach_loader_selection_without_file_count_dilution(self):
        originals = {p: p.read_bytes() for p in self.author_files + self.grail_files}
        report = self.prepare()
        manifest = yaml.safe_load(Path(report["selection"]).read_text(encoding="utf-8"))
        # Round-trip through the existing input validator, not just report fields.
        loaded = validate_selection(report["root"], report["selection"])
        self.assertEqual(set(manifest["selected_files"]), {p.as_posix() for p in originals})
        self.assertEqual(loaded["files"], 7)
        weights = {m["path"]: m["weight"] for m in report["motions"]}
        self.assertAlmostEqual(weights[self.author_files[0].as_posix()], 0.375)
        self.assertAlmostEqual(weights[self.author_files[1].as_posix()], 0.125)
        self.assertAlmostEqual(sum(weights[p.as_posix()] for p in self.grail_files), 0.5)
        self.assertAlmostEqual(sum(manifest["motion_weights"]), 1.)
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)

    def test_missing_grail_subset_does_not_silently_train_on_authors(self):
        self.grail_files[1].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "slope"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_source_overlap_is_rejected(self):
        selection = self.root / "overlap.yaml"
        selection.write_text(yaml.safe_dump({
            "selected_files": [p.as_posix() for p in self.grail_files + self.author_files[:1]],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same motion occurs in both"):
            self.prepare(grail_selection=selection)

    def test_resume_rejects_changed_source_mass(self):
        report = self.prepare()
        params = self.root / "run/params"
        params.mkdir(parents=True)
        (params / "motion_inventory.json").write_text(json.dumps(report), encoding="utf-8")
        checkpoint = params.parent / "model_1.pt"
        check_resume_motion_inventory(checkpoint, self.prepare())
        with self.assertRaisesRegex(ValueError, "references/weights differ"):
            check_resume_motion_inventory(checkpoint, self.prepare(grail_fraction=0.6))


if __name__ == "__main__":
    unittest.main()
