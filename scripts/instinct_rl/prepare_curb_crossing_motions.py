"""Extract kinematic crossing candidates from converted GRAIL curb motions.

No simulator, GPU, download or source-data edits. Elevated low-velocity sole
intervals are only support proxies: without terrain assets/contact labels this
filter cannot certify that two supports belong to different curb platforms.
The training task supplies the actual platform-to-platform contact objective.
"""

import argparse
import hashlib
import json
import math
import numpy as np
import sys
import tempfile
import yaml
from pathlib import Path

from check_parkour_motions import ROBOT, ROOT, filesystem_path, sha256_motion, validate_selection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foot_state_demo"))
from foot_state import LegKinematics, quaternion_matrix


def bridge_windows(sole, rate):
    """Return frame windows around alternating, elevated, forward support proxies.

    Reject windows containing a low-ground stance, a large height change or a
    turn/backward step. Crop the original trajectory; do not synthesize poses.
    """
    if not math.isfinite(rate) or rate <= 0 or len(sole) < 4:
        raise ValueError("Need finite positive framerate and >=4 sole samples")
    speed = np.linalg.norm(np.gradient(sole, 1 / rate, axis=0), axis=-1)
    stationary = speed < 0.15
    if not stationary.any():
        return []
    floor = float(np.percentile(sole[..., 2][stationary], 5))
    runs = []
    for foot in range(2):
        boundaries = np.flatnonzero(np.diff(np.r_[False, stationary[:, foot], False].astype(int)))
        for start, end in zip(boundaries[::2], boundaries[1::2]):
            if end - start >= max(2, math.ceil(0.12 * rate)):
                runs.append((int(start), int(end), foot, np.median(sole[start:end, foot], axis=0)))
    runs.sort(key=lambda item: item[0])
    windows = []
    for i, second in enumerate(runs):
        # The other foot can remain planted while this foot completes several
        # stance/swing phases. Adjacent entries in the run list are insufficient.
        predecessors = [
            r for r in runs[:i] if r[2] != second[2] and r[0] < second[0] and r[1] >= second[0] - math.ceil(0.10 * rate)
        ]
        if not predecessors:
            continue
        first = predecessors[-1]
        start, end, foot, p = first
        next_start, next_end, next_foot, target = second
        delta = target - p
        distance = np.linalg.norm(delta[:2])
        if (
            foot == next_foot
            or not 0.40 <= distance <= 0.70
            or min(p[2], target[2]) < floor + 0.06
            or abs(delta[2]) > 0.24
        ):
            continue
        # Express displacement along the local motion direction, including
        # reverse traversals, while rejecting sideways steps and U-turns.
        midpoint = sole.mean(axis=1)
        direction = midpoint[min(next_end, len(sole) - 1), :2] - midpoint[max(0, start - 1), :2]
        norm = np.linalg.norm(direction)
        if norm < 0.05 or np.dot(delta[:2], direction) < 0.70 * distance * norm:
            continue
        if any(r[0] < next_end and r[1] > start and r[3][2] < floor + 0.04 for r in runs):
            continue
        # Boundaries remain inside elevated support intervals so ground
        # approach/exit frames do not dominate the new AMP corpus.
        if next_end - start >= math.ceil(0.30 * rate):
            window = (start, next_end)
            if not windows or window[0] >= windows[-1][1]:
                windows.append(window)
    return windows


def prepare_references(source_root, output_root, robot_path=ROBOT):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    files = sorted((source_root / "curb").glob("*_retargeted.npz"))
    if not files:
        raise FileNotFoundError("No converted GRAIL curb NPZ files under " + str(source_root / "curb"))
    # Validate every source and its G1 mapping before extracting any windows.
    source_signature = [(p.name, sha256_motion(p)) for p in files]
    robot_hash = hashlib.sha256(Path(robot_path).read_bytes()).hexdigest()
    provenance_path = output_root / "curation.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            old.get("source_signature") != [list(item) for item in source_signature]
            or old.get("robot_sha256") != robot_hash
        ):
            raise ValueError("Source motions/URDF changed; use a new --output_root instead of overwriting references")
        inventory = validate_selection(output_root, output_root / "crossing.yaml", robot_path)
        if [(m["path"], m["sha256"]) for m in inventory["motions"]] != [
            tuple(item) for item in old["output_signature"]
        ]:
            raise ValueError("Prepared references changed; use a new output directory")
        return inventory
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Output directory must be empty or contain a complete matching curation.json")
    records, selected, kinematics = [], [], None
    # Validate the source without needing all four Mixed reference subsets.
    with tempfile.TemporaryDirectory() as directory:
        source_selection = Path(directory) / "source_selection.yaml"
        source_selection.write_text(
            yaml.safe_dump({"selected_files": ["curb/" + p.name for p in files]}), encoding="utf-8"
        )
        validate_selection(source_root, source_selection, robot_path)
    for path, (_, source_hash) in zip(files, source_signature):
        with np.load(filesystem_path(path), allow_pickle=True) as clip:
            names = clip["joint_names"].tolist()
            if kinematics is None or kinematics.joint_names != tuple(names):
                kinematics = LegKinematics(robot_path, names)
            if kinematics.root_link != "torso_link":
                raise ValueError("Curb references must use the existing torso-root G1 convention")
            q = clip["joint_pos"]
            rate = float(clip["framerate"].item())
            p, _, _ = kinematics.compute(q, np.zeros_like(q))
            world = np.einsum("nij,nfj->nfi", quaternion_matrix(clip["base_quat_w"]), p)
            world += clip["base_pos_w"][:, None]
            for start, end in bridge_windows(world, rate):
                # Short numeric names avoid Windows' long GRAIL file paths.
                relative = f"curb/{source_hash[:16]}_{start}_{end}_retargeted.npz"
                destination = output_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    destination,
                    framerate=rate,
                    joint_names=clip["joint_names"],
                    joint_pos=q[start:end],
                    base_pos_w=clip["base_pos_w"][start:end],
                    base_quat_w=clip["base_quat_w"][start:end],
                )
                selected.append(relative)
                records.append(
                    dict(
                        source="curb/" + path.name,
                        sha256=source_hash,
                        start_frame=start,
                        end_frame_exclusive=end,
                        output=relative,
                    )
                )
    if not selected:
        raise ValueError(
            "No elevated crossing candidates found. Supply a reviewed --motion_selection; no fallback is used"
        )
    selection = output_root / "crossing.yaml"
    selection.write_text(
        yaml.safe_dump({"selected_files": selected, "motion_weights": [1.0] * len(selected)}), encoding="utf-8"
    )
    inventory = validate_selection(output_root, selection, robot_path)
    provenance_path.write_text(
        json.dumps(
            dict(
                version=1,
                source_signature=source_signature,
                robot_sha256=robot_hash,
                windows=records,
                output_signature=[(m["path"], m["sha256"]) for m in inventory["motions"]],
                scope=(
                    "Kinematic candidates, not contact/terrain-certified bridges; inspect against source demonstrations"
                ),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return inventory


def configure_curb_motions(env_cfg, motion_root=None, motion_selection=None):
    motion = env_cfg.scene.motion_reference.motion_buffers["run_walk"]
    root = Path(motion_root or motion.path).expanduser().resolve()
    selection = motion_selection or motion.filtered_motion_selection_filepath
    if selection:
        inventory = validate_selection(root, selection, env_cfg.scene.robot.spawn.asset_path)
    else:
        cache_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
        inventory = prepare_references(
            root, ROOT / "outputs/curb_crossing_references" / cache_key, env_cfg.scene.robot.spawn.asset_path
        )
    if any(not m["path"].startswith("curb/") for m in inventory["motions"]):
        raise ValueError("Curb crossing selection must contain only converted GRAIL curb references under curb/")
    motion.path = inventory["root"]
    motion.filtered_motion_selection_filepath = inventory["selection"]
    motion.subset_selection, motion.file_path_patterns = None, None
    inventory["reference_scope"] = "GRAIL curb kinematic crossing candidates or explicitly reviewed selection"
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=ROOT / "data/grail_instinctlab")
    parser.add_argument("--output_root", type=Path, required=True)
    options = parser.parse_args()
    report = prepare_references(options.source_root, options.output_root)
    print(json.dumps({key: value for key, value in report.items() if key != "motions"}, indent=2))
