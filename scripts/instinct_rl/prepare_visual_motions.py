"""Combine author walking and converted GRAIL motions for one AMP policy.

Only the generated selection is written; source motion files are not copied or
modified. This preparation and validation do not import Isaac Sim or PyTorch.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import yaml

from check_parkour_motions import AUTHOR_ROOT, AUTHOR_SELECTION, ROBOT, ROOT, validate_selection
from prepare_parkour_references import prepare_references, write_atomic


def prepare_visual_motions(
    grail_root, grail_selection=None, author_root=None, grail_fraction=0.5, robot_path=ROBOT, output_dir=None
):
    """Validate both corpora and give each source an explicit sampling mass.

    Fractions refer to initial clip selection, not frame counts or terrain
    assignments. Within each source, preserve its relative motion weights.
    """
    if not math.isfinite(grail_fraction) or not 0 < grail_fraction < 1:
        raise ValueError("grail_fraction must be strictly between 0 and 1; both corpora must participate.")
    grail = validate_selection(grail_root, grail_selection, robot_path)
    author_root = Path(
        author_root or os.environ.get("INSTINCTLAB_AUTHOR_MOTION_ROOT") or AUTHOR_ROOT
    ).expanduser().resolve()
    provenance = prepare_references() if author_root == AUTHOR_ROOT.resolve() else None
    author = validate_selection(author_root, author_root / AUTHOR_SELECTION, robot_path)
    records, sources, subset_counts, subset_probabilities = [], {}, Counter(), Counter()
    selected_files, seen = [], set()
    for name, report, mass in (("author", author, 1 - grail_fraction), ("grail", grail, grail_fraction)):
        total = math.fsum(m["weight"] for m in report["motions"])
        sources[name] = {key: report[key] for key in ("root", "selection", "selection_sha256", "files")}
        sources[name]["initial_clip_probability"] = mass
        for item in report["motions"]:
            path = (Path(report["root"]) / item["path"]).resolve()
            if path in seen:
                raise ValueError(f"The same motion occurs in both expert sources: {path}")
            seen.add(path)
            # Absolute paths also support datasets on different Windows drives.
            # The existing AmassMotion YAML loader accepts these via os.path.join.
            selected_files.append(path.as_posix())
            weight = mass * item["weight"] / total
            subset = "author" if name == "author" else item["path"].split("/")[0]
            subset_counts[subset] += 1
            subset_probabilities[subset] += weight
            records.append(dict(item, path=path.as_posix(), weight=weight, source=name, subset=subset))
    payload = yaml.safe_dump(
        {"selected_files": selected_files, "motion_weights": [m["weight"] for m in records]},
        allow_unicode=False,
        sort_keys=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(output_dir or ROOT / "outputs/visual_foothold/reference_manifests").expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    selection = directory / f"author_grail_{digest}.yaml"
    write_atomic(selection, payload)
    return dict(
        root=str(ROOT),
        selection=str(selection),
        selection_sha256=digest,
        reference_role="AMP expert distribution; not time-aligned motion tracking",
        files=len(records),
        frames=author["frames"] + grail["frames"],
        duration_s=author["duration_s"] + grail["duration_s"],
        checked_files=len(records),
        subset_counts=dict(subset_counts),
        initial_subset_probability=dict(subset_probabilities),
        sources=sources,
        initial_source_probability={"author": 1 - grail_fraction, "grail": grail_fraction},
        author_reference_provenance=provenance,
        validation=grail["validation"],
        motions=records,
    )


def configure_visual_motions(env_cfg, motion_root=None, motion_selection=None, author_root=None, grail_fraction=0.5):
    """Install the joint expert pool before gym.make, preserving both sources."""
    reference = env_cfg.scene.motion_reference
    motion = reference.motion_buffers["run_walk"]
    root = motion_root or motion.path
    # A root override must not retain a manifest from the previous root.
    selection = motion_selection or (None if motion_root else motion.filtered_motion_selection_filepath)
    report = prepare_visual_motions(root, selection, author_root, grail_fraction, env_cfg.scene.robot.spawn.asset_path)
    motion.path = report["root"]
    motion.filtered_motion_selection_filepath = report["selection"]
    motion.subset_selection = None
    motion.file_path_patterns = None
    motion.env_starting_stub_sampling_strategy = "independent"
    # Sharding 1501 clips could leave the sole author clip on only one rank.
    # Keep the weighted joint corpus available in every training process.
    reference.mp_split_method = "None"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion_root",
        default=os.environ.get("INSTINCTLAB_GRAIL_MOTION_ROOT") or str(ROOT / "data/grail_instinctlab"),
        help="Converted GRAIL root",
    )
    parser.add_argument("--motion_selection", help="Optional selection within the GRAIL source")
    parser.add_argument("--author_motion_root", default=os.environ.get("INSTINCTLAB_AUTHOR_MOTION_ROOT"))
    parser.add_argument("--amp_grail_fraction", type=float, default=0.5)
    parser.add_argument("--output_dir", help="Directory for the generated joint selection")
    args = parser.parse_args()
    try:
        report = prepare_visual_motions(
            args.motion_root, args.motion_selection, args.author_motion_root,
            args.amp_grail_fraction, output_dir=args.output_dir,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"[VISUAL AMP] ERROR: {error}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "motions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
