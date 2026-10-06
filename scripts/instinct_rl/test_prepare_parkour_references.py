"""CPU tests for author reference integrity, offline installation and reuse."""

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import prepare_parkour_references as prepare
from check_parkour_motions import ROBOT


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target = self.root / "installed"
        names = [j.attrib["name"] for j in ET.parse(ROBOT).getroot().findall("joint") if j.attrib["type"] != "fixed"]
        clip = self.root / "reference.npz"
        np.savez(clip, framerate=50., joint_names=np.array(names), joint_pos=np.zeros((10, len(names))),
                 base_pos_w=np.zeros((10, 3)), base_quat_w=np.tile([1., 0., 0., 0.], (10, 1)))
        data = {
            "parkour_motion_without_run_retargetted.npz": clip.read_bytes(),
            prepare.SELECTION_NAME: b"selected_files:\n- parkour_motion_without_run_retargetted.npz\n",
        }
        files = {name: (prepare.FILES[name][0], hashlib.sha256(payload).hexdigest()) for name, payload in data.items()}
        self.archive = self.root / "author.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, payload in data.items():
                archive.writestr(files[name][0], payload)
            archive.writestr("data&model/README.md", "fixture readme")
            archive.writestr("../escape.txt", "must never extract")
        self.addCleanup(patch.stopall)
        patch.object(prepare, "FILES", files).start()
        patch.object(prepare, "ARCHIVE_SHA256", prepare.sha256_file(self.archive)).start()

    def test_offline_install_copies_original_manifest_and_only_fixed_members(self):
        report = prepare.prepare_references(self.target, self.archive)
        self.assertEqual(report["frames"], 10)
        self.assertEqual(report["duration_s"], .18)
        self.assertFalse((self.root / "escape.txt").exists())
        self.assertEqual({p.name for p in self.target.iterdir()}, set(prepare.FILES) | {"AUTHOR_README.md", "provenance.json"})
        self.assertEqual(json.loads((self.target / "provenance.json").read_text())["online_mpc"], False)

    def test_reuse_is_offline_and_does_not_download_again(self):
        prepare.prepare_references(self.target, self.archive)
        with patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("should be offline")):
            prepare.prepare_references(self.target)

    def test_archive_hash_mismatch_writes_nothing(self):
        self.archive.write_bytes(b"incomplete")
        with self.assertRaisesRegex(ValueError, "archive SHA256 mismatch"):
            prepare.prepare_references(self.target, self.archive)
        self.assertFalse(self.target.exists())

    def test_changed_existing_data_is_preserved_and_rejected(self):
        self.target.mkdir()
        existing = self.target / prepare.SELECTION_NAME
        existing.write_bytes(b"my selection")
        with self.assertRaisesRegex(ValueError, "Existing reference differs"):
            prepare.prepare_references(self.target, self.archive)
        self.assertEqual(existing.read_bytes(), b"my selection")

    def test_wrong_g1_mapping_fails_before_training(self):
        # A verified archive still must be compatible with the training robot.
        with patch("check_parkour_motions.validate_selection", side_effect=ValueError("G1 joint mapping mismatch")):
            with self.assertRaisesRegex(ValueError, "joint mapping mismatch"):
                prepare.prepare_references(self.target, self.archive)
        self.assertFalse((self.target / "provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
