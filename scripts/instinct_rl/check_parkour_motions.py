"""Check mixed Parkour motion selection without importing Isaac Sim or PyTorch.

Checks file availability, sampling weights and every clip's G1 data format.
It cannot establish gait quality or infer walking style from a directory name.
"""

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOT = ROOT / "source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
SUBSETS = ("curb", "slope", "stair_p1", "stair_p2")
PATTERNS = [f"{subset}/*_retargeted.npz" for subset in SUBSETS]


def filesystem_path(path):
    """Permit the long GRAIL clip names on Windows as well as Linux."""
    name = os.path.abspath(path)
    if os.name != "nt" or name.startswith("\\\\?\\"):
        return name
    return "\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name


def validate_selection(root, selection=None, robot_path=ROBOT):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Motion root not found: {root}. Pass --motion_root or set INSTINCTLAB_PARKOUR_MOTION_ROOT.")
    selection = Path(selection).expanduser().resolve() if selection else None
    if selection:
        data = yaml.safe_load(selection.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("selected_files"), list) or not data["selected_files"]:
            raise ValueError("Selection YAML must contain a non-empty selected_files list.")
        if any(not isinstance(name, str) or not name for name in data["selected_files"]):
            raise ValueError("selected_files entries must be non-empty paths.")
        files = [(root / name).resolve() for name in data["selected_files"]]
        # This is the key actually read by AmassMotion._refresh_motion_file_list.
        if "weights" in data:
            raise ValueError("Use motion_weights, not weights: the installed AmassMotion loader reads motion_weights.")
        weights = data.get("motion_weights", [1.0] * len(files))
    else:
        files = []
        for subset, pattern in zip(SUBSETS, PATTERNS):
            found = sorted(root.glob(pattern))
            if not found:
                raise FileNotFoundError(
                    f"No {pattern} under {root}. Mixed training requires all four subsets; "
                    "it will not fall back to stairs-only data. Alternatively supply --motion_selection with curated G1 motions."
                )
            files.extend(found)
        weights = [1.0] * len(files)
    if len(set(files)) != len(files):
        raise ValueError("Duplicate motion files in selection.")
    if not isinstance(weights, list) or len(weights) != len(files):
        raise ValueError("motion_weights must have one weight per selected file.")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("motion_weights must be finite non-negative numbers.")
    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("motion_weights must have a finite positive sum.")

    expected = {joint.attrib["name"] for joint in ET.parse(robot_path).getroot().findall("joint")
                if joint.attrib.get("type") != "fixed"}
    records, subset_counts, subset_weights, total_frames = [], Counter(), Counter(), 0
    for path, weight in zip(files, weights):
        if not os.path.isfile(filesystem_path(path)):
            raise FileNotFoundError(f"Selected motion missing: {path}")
        if not str(path).endswith(("retargeted.npz", "retargetted.npz")):
            raise ValueError(f"This task needs G1 retargeted NPZ files, not raw SMPL data: {path}")
        # Trusted local motion files use the same joint_names representation as
        # the existing AmassMotion loader (which also enables allow_pickle).
        with np.load(filesystem_path(path), allow_pickle=True) as clip:
            required = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
            if not required.issubset(clip.files):
                raise ValueError(f"Missing {sorted(required - set(clip.files))} in {path}")
            names = clip["joint_names"].tolist()
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError(f"Invalid joint_names in {path}")
            if len(set(names)) != len(names) or not expected.issubset(names):
                raise ValueError(f"G1 joint mapping mismatch in {path}; missing {sorted(expected - set(names))}")
            joints, pos, quat = clip["joint_pos"], clip["base_pos_w"], clip["base_quat_w"]
            rate = float(clip["framerate"].item())
            if joints.ndim != 2 or joints.shape[0] < 2 or joints.shape[1] != len(names):
                raise ValueError(f"Invalid joint_pos shape in {path}: {joints.shape}")
            n = joints.shape[0]
            if pos.shape != (n, 3) or quat.shape != (n, 4) or not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"Invalid root trajectory or framerate in {path}")
            if not all(np.isfinite(array).all() for array in (joints, pos, quat)):
                raise ValueError(f"Non-finite motion data in {path}")
            if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=0.01):
                raise ValueError(f"Non-unit base quaternion in {path}")
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        subset = relative.split("/")[0] if "/" in relative else "root"
        subset_counts[subset] += 1
        subset_weights[subset] += weight / total_weight
        total_frames += n
        records.append(dict(path=relative, frames=n, framerate=rate, weight=weight))
    # Known stairs-only selections must not reproduce the old training mistake.
    active_subsets = {name for name, probability in subset_weights.items() if probability > 0}
    if active_subsets and active_subsets <= {"stair_p1", "stair_p2"}:
        raise ValueError("Selection contains only active GRAIL stair references. Include reviewed locomotion/curb/slope references.")
    return dict(
        root=str(root), selection=str(selection) if selection else None,
        selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest() if selection else None,
        files=len(files), frames=total_frames, subset_counts=dict(subset_counts),
        initial_subset_probability=dict(subset_weights), checked_files=len(records),
        validation="availability, weights, G1 joint mapping, shapes, finite values, unit quaternions; not gait quality",
        motions=records,
    )


def configure_training_motions(env_cfg, motion_root=None, motion_selection=None):
    motion = env_cfg.scene.motion_reference.motion_buffers["run_walk"]
    if motion_root:
        motion.path = str(Path(motion_root).expanduser().resolve())
    if motion_selection:
        motion.filtered_motion_selection_filepath = str(Path(motion_selection).expanduser().resolve())
    motion.subset_selection = None
    motion.file_path_patterns = None if motion.filtered_motion_selection_filepath else PATTERNS.copy()
    return validate_selection(motion.path, motion.filtered_motion_selection_filepath, env_cfg.scene.robot.spawn.asset_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_root", default=os.environ.get("INSTINCTLAB_PARKOUR_MOTION_ROOT", os.environ.get("INSTINCTLAB_GRAIL_MOTION_ROOT")))
    parser.add_argument("--motion_selection", default=os.environ.get("INSTINCTLAB_PARKOUR_MOTION_SELECTION"))
    parser.add_argument("--scan", action="store_true", help="List known dataset locations and original selection manifests")
    args = parser.parse_args()
    candidates = [Path(args.motion_root).expanduser()] if args.motion_root else [
        Path("/workspace/instinctlab/data/grail_instinctlab"), ROOT / "data/grail_instinctlab", Path.home() / "Datasets",
    ]
    if args.scan:
        for root in dict.fromkeys(Path(os.path.abspath(path)) for path in candidates):
            print(json.dumps(dict(root=str(root), exists=root.is_dir(), subsets={
                name: len(list(root.glob(f"{name}/*_retargeted.npz"))) for name in SUBSETS
            }, original_selection=str(root / "parkour_motion_without_run.yaml")
                if (root / "parkour_motion_without_run.yaml").is_file() else None), ensure_ascii=False))
        return
    root = next((path for path in candidates if path.is_dir()), candidates[0])
    try:
        report = validate_selection(root, args.motion_selection)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"[MOTIONS] ERROR: {error}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "motions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
