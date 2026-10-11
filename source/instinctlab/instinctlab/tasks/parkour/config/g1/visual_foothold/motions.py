"""Locate the GRAIL input to the joint author + GRAIL AMP expert pool."""

import copy
import os
from pathlib import Path


GRAIL_ROOT = Path(__file__).resolve().parents[8] / "data/grail_instinctlab"
GRAIL_SUBSETS = ("curb", "slope", "stair_p1", "stair_p2")


def grail_motion_reference(reference):
    """Set the GRAIL input root without modifying the original Mixed task.

    Use converted G1 NPZs as an unconditional AMP expert pool, not as terrain
    labels or time-aligned foothold targets. Before gym.make, train.py calls
    configure_visual_motions to combine this source with the retained author
    corpus, then records the final joint selection and sampling weights.
    """
    reference = copy.deepcopy(reference)
    motion = reference.motion_buffers["run_walk"]
    root = (os.environ.get("INSTINCTLAB_PARKOUR_MOTION_ROOT")
            or os.environ.get("INSTINCTLAB_GRAIL_MOTION_ROOT") or str(GRAIL_ROOT))
    selection = os.environ.get("INSTINCTLAB_PARKOUR_MOTION_SELECTION")
    motion.path = os.path.expanduser(root)
    # Clear the author's YAML inherited from Mixed, including when GRAIL is
    # missing. Missing references must fail validation, never fall back silently.
    motion.filtered_motion_selection_filepath = os.path.expanduser(selection) if selection else None
    motion.subset_selection = None
    motion.file_path_patterns = None if selection else [f"{subset}/*_retargeted.npz" for subset in GRAIL_SUBSETS]
    return reference
