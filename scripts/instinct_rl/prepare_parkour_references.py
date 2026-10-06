"""Prepare the authors' offline MPC + mocap walking reference for G1 AMP.

No Isaac Sim, ROS, online MPC solver, or motion retargeting is required here.
The original NPZ and selection YAML are copied byte-for-byte from the authors'
public Data&Model archive. Only these two files and its README are extracted.
"""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data/hiking_in_the_wild/parkour_motion_reference"
SELECTION_NAME = "parkour_motion_without_run.yaml"
PROJECT_URL = "https://project-instinct.github.io/hiking-in-the-wild/"
DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=10tQylYHdKLVDVnmVHrygLB70neHnPdoF&export=download&confirm=t"
)
ARCHIVE_SIZE = 11905690
ARCHIVE_SHA256 = "bbba90bdb77bbab19e74cbe9f556ea0b9330ea9d125ad8271d431aa0e37a6609"
FILES = {
    "parkour_motion_without_run_retargetted.npz": (
        "data&model/parkour_motion_reference/parkour_motion_without_run_retargetted.npz",
        "7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b",
    ),
    SELECTION_NAME: (
        "data&model/parkour_motion_reference/parkour_motion_without_run.yaml",
        "f79e5bbc9207976e1610459ab3727a9e1da6d5c0c6cc75793dcec34b81cb7679",
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path, data):
    """Avoid leaving a partial reference/manifest after interruption."""
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False, suffix=".tmp") as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_archive(archive, root):
    """Accept the pinned author archive and extract fixed members only."""
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("Author archive SHA256 mismatch. Refusing changed or incomplete data.")
    # Read/check all members before writing anything. Never use extractall.
    with zipfile.ZipFile(archive) as source:
        payloads = {}
        for filename, (member, expected_hash) in FILES.items():
            data = source.read(member)
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError("Author reference SHA256 mismatch: " + filename)
            payloads[filename] = data
        payloads["AUTHOR_README.md"] = source.read("data&model/README.md")
    root.mkdir(parents=True, exist_ok=True)
    for filename, data in payloads.items():
        write_atomic(root / filename, data)


def prepare_references(root=DEFAULT_ROOT, archive=None):
    root = Path(root).expanduser().resolve()
    for filename, (_, expected_hash) in FILES.items():
        target = root / filename
        if target.exists() and sha256_file(target) != expected_hash:
            raise ValueError(
                "Existing reference differs from the pinned author data: " + str(target)
                + ". Use a separate --output_dir or select your custom data with --motion_root/--motion_selection."
            )
    ready = all((root / name).is_file() for name in FILES)
    if not ready:
        if archive:
            extract_archive(Path(archive).expanduser().resolve(), root)
        else:
            # Use a private temporary file; concurrent preparations cannot share
            # a partially downloaded archive. Range also avoids Drive previews.
            print("[REFERENCES] Downloading authors' G1 walking data (11.9 MB)...", flush=True)
            try:
                with tempfile.TemporaryDirectory() as directory:
                    downloaded = Path(directory) / "author_data_model.zip"
                    request = urllib.request.Request(DOWNLOAD_URL, headers={
                        "User-Agent": "InstinctLab-reference-preparation",
                        "Range": "bytes=0-{}".format(ARCHIVE_SIZE - 1),
                    })
                    with urllib.request.urlopen(request, timeout=30) as response, open(downloaded, "wb") as stream:
                        size = 0
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > ARCHIVE_SIZE:
                                raise ValueError("Author archive exceeds its pinned size.")
                            stream.write(chunk)
                    if size != ARCHIVE_SIZE:
                        raise ValueError("Incomplete author archive: {} of {} bytes".format(size, ARCHIVE_SIZE))
                    extract_archive(downloaded, root)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                raise RuntimeError(
                    "Cannot prepare author references: {}. For an offline server, download Data&Model from {} "
                    "and run python scripts/instinct_rl/prepare_parkour_references.py --archive /path/to/DataModel.zip."
                    .format(error, PROJECT_URL)
                ) from error
    # Use the same schema and joint mapping checks as training, after SHA checks.
    from check_parkour_motions import validate_selection

    inventory = validate_selection(root, root / SELECTION_NAME)
    provenance = dict(
        project_url=PROJECT_URL, download_url=DOWNLOAD_URL, archive_sha256=ARCHIVE_SHA256,
        reference_kind="author walking corpus; paper describes offline MPC + NOKOV/GMR",
        online_mpc=False, file_sha256={name: values[1] for name, values in FILES.items()},
        frames=inventory["frames"], duration_s=inventory["duration_s"],
        clip_source_labels_available=False,
    )
    write_atomic(root / "provenance.json", (json.dumps(provenance, indent=2) + "\n").encode("utf-8"))
    print("[REFERENCES] Verified {} frames, {:.2f} s; selection={}".format(
        provenance["frames"], provenance["duration_s"], root / SELECTION_NAME), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Offline copy of the official Data&Model ZIP")
    parser.add_argument("--output_dir", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    try:
        prepare_references(args.output_dir, args.archive)
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, "[REFERENCES] ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
