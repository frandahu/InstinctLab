"""Upstream Parkour recipe with the available multi-terrain G1 motion data.

Only reference selection differs from upstream. Rewards, terrain proportions,
curriculum, sensors, commands, resets, and joint control come from G1ParkourEnvCfg.
"""

import copy
import os

from isaaclab.utils import configclass

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY


def mixed_motion_reference(reference):
    reference = copy.deepcopy(reference)
    motion = reference.motion_buffers["run_walk"]
    motion.path = os.path.expanduser(os.environ.get(
        "INSTINCTLAB_PARKOUR_MOTION_ROOT",
        os.environ.get("INSTINCTLAB_GRAIL_MOTION_ROOT", "/workspace/instinctlab/data/grail_instinctlab"),
    ))
    selection = os.environ.get("INSTINCTLAB_PARKOUR_MOTION_SELECTION")
    motion.filtered_motion_selection_filepath = os.path.expanduser(selection) if selection else None
    motion.subset_selection = None
    # No implicit stairs-only fallback. A curated YAML can replace this selection.
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


@configclass
class G1MixedParkourEnvCfg_PLAY(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.motion_reference = mixed_motion_reference(self.scene.motion_reference)
