"""Upstream Parkour recipe with the authors' MPC + mocap walking reference.

Reference selection and feet_air_time weight differ from upstream. Other rewards,
terrain proportions, curriculum, sensors, commands, resets, and joint control
come from G1ParkourEnvCfg.
"""

import copy
import os
from pathlib import Path

from isaaclab.utils import configclass

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY


def mixed_motion_reference(reference):
    reference = copy.deepcopy(reference)
    motion = reference.motion_buffers["run_walk"]
    default_root = Path(__file__).resolve().parents[7] / "data/hiking_in_the_wild/parkour_motion_reference"
    root_override = os.environ.get("INSTINCTLAB_PARKOUR_MOTION_ROOT")
    motion.path = os.path.expanduser(root_override) if root_override else str(default_root)
    selection = os.environ.get("INSTINCTLAB_PARKOUR_MOTION_SELECTION")
    if not selection and not root_override:
        selection = str(default_root / "parkour_motion_without_run.yaml")
    motion.filtered_motion_selection_filepath = os.path.expanduser(selection) if selection else None
    motion.subset_selection = None
    # Custom roots can still explicitly select the converted GRAIL subsets.
    motion.file_path_patterns = None if selection else [
        "curb/*_retargeted.npz", "slope/*_retargeted.npz",
        "stair_p1/*_retargeted.npz", "stair_p2/*_retargeted.npz",
    ]
    return reference


@configclass
class G1MixedParkourEnvCfg(G1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference = mixed_motion_reference(self.scene.motion_reference)
        self.rewards.rewards.feet_air_time.weight = 0.75


@configclass
class G1MixedParkourEnvCfg_PLAY(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference = mixed_motion_reference(self.scene.motion_reference)
        self.rewards.rewards.feet_air_time.weight = 0.75
