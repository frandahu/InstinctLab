"""CPU-only contracts for task-local walls and legacy saved-terrain playback."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as Config
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/instinctlab/instinctlab/tasks/parkour/config/g1/visual_foothold/terrain.py"
spec = importlib.util.spec_from_file_location("visual_terrain_cpu", SOURCE)
terrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terrain)


def original_generator():
    return Config(
        size=(8., 8.), num_rows=10, num_cols=20, border_width=3., border_height=1.,
        sub_terrains={
            "stairs": Config(proportion=.6, step_height_range=(.05, .45), step_width=.3,
                             wall_prob=[.3]*4, wall_height=5., wall_thickness=.05,
                             perlin_cfg=Config(noise_scale=.05, wall_prob=[.1]*4)),
            "boxes": Config(proportion=.4, box_height_mean=[.1, .4], wall_prob=[.3]*4,
                            wall_height=5., wall_thickness=.05),
        },
    )


class TerrainWallContracts(unittest.TestCase):
    def test_wall_removal_is_isolated_and_preserves_complex_surfaces(self):
        original = original_generator()
        fixed = terrain.configure_terrain_walls(original, "off")
        self.assertFalse(terrain.terrain_wall_summary(fixed)["random_walls_enabled"])
        self.assertEqual(original.sub_terrains["stairs"].wall_prob, [.3]*4)
        self.assertEqual(original.sub_terrains["stairs"].perlin_cfg.wall_prob, [.1]*4)
        self.assertEqual(original.sub_terrains["boxes"].wall_prob, [.3]*4)
        self.assertEqual(fixed.sub_terrains["stairs"].step_height_range, (.05, .45))
        self.assertEqual(fixed.sub_terrains["stairs"].step_width, .3)
        self.assertEqual(fixed.sub_terrains["stairs"].perlin_cfg.noise_scale, .05)
        self.assertEqual(fixed.sub_terrains["boxes"].box_height_mean, [.1, .4])
        self.assertEqual(fixed.size, (8., 8.))
        self.assertEqual((fixed.border_width, fixed.border_height), (3., 1.))
        self.assertEqual([s.proportion for s in fixed.sub_terrains.values()], [.6, .4])

    def test_restored_wall_settings_are_overridden_or_reproduced_explicitly(self):
        # This is the state AFTER env.yaml restoration: wall-free task defaults
        # have already been overwritten by the historical 30% probabilities.
        restored = original_generator()
        off = terrain.configure_terrain_walls(restored, "off")
        replay = terrain.configure_terrain_walls(restored)
        self.assertFalse(terrain.terrain_wall_summary(off)["random_walls_enabled"])
        self.assertEqual(replay.sub_terrains["boxes"].wall_prob, [.3]*4)
        self.assertTrue(terrain.terrain_wall_summary(replay)["random_walls_enabled"])
        self.assertIsNot(replay.sub_terrains["boxes"], restored.sub_terrains["boxes"])
        with self.assertRaisesRegex(ValueError, "wall mode"):
            terrain.configure_terrain_walls(restored, "unknown")

    def test_saved_mapping_and_live_config_audits_are_json_serializable(self):
        restored = {
            "size": (8., 8.), "num_rows": 10, "num_cols": 20, "border_width": 3., "border_height": 1.,
            "sub_terrains": {"stairs": {"wall_prob": [.3]*4, "wall_height": 5., "wall_thickness": .05}},
        }
        report = terrain.terrain_wall_summary(restored)
        self.assertEqual(report["wall_enabled_configs"], ["stairs"])
        self.assertEqual(report["wall_configs"]["stairs"]["height_m"], 5.)
        off = terrain.configure_terrain_walls(restored, "off")
        self.assertEqual(restored["sub_terrains"]["stairs"]["wall_prob"], [.3]*4)
        audit = {"restored": report, "effective": terrain.terrain_wall_summary(off)}
        encoded = json.loads(json.dumps(audit, allow_nan=False))
        self.assertFalse(encoded["effective"]["random_walls_enabled"])
        self.assertTrue(encoded["restored"]["random_walls_enabled"])
        self.assertEqual(terrain.terrain_wall_summary(original_generator())["size_m"], [8., 8.])


if __name__ == "__main__":
    unittest.main()
