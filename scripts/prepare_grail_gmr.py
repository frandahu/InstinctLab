import os
import pickle
import joblib


BASE = "/workspace/instinctlab"

RAW_ROOT = os.path.join(
    BASE,
    "data/grail_raw",
)

MANIFEST_ROOT = os.path.join(
    RAW_ROOT,
    "manifests",
)

OUTPUT_ROOT = os.path.join(
    BASE,
    "data/grail_gmr_compatible",
)

CATEGORIES = [
    "curb",
    "slope",
    "stair_p1",
    "stair_p2",
]


total = 0

for category in CATEGORIES:

    manifest_path = os.path.join(
        MANIFEST_ROOT,
        f"{category}.txt",
    )

    output_dir = os.path.join(
        OUTPUT_ROOT,
        category,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    with open(manifest_path, "r") as f:
        repo_files = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print()
    print("=" * 60)
    print(f"{category}: {len(repo_files)} files")
    print("=" * 60)

    for i, repo_path in enumerate(repo_files, 1):

        src = os.path.join(
            RAW_ROOT,
            repo_path,
        )

        raw = joblib.load(src)

        if not isinstance(raw, dict):
            raise TypeError(
                f"Unexpected object in {src}: {type(raw)}"
            )

        if len(raw) != 1:
            raise RuntimeError(
                f"Unexpected top-level structure in {src}"
            )

        motion_name = next(iter(raw))
        d = raw[motion_name]

        required = [
            "dof",
            "root_trans_offset",
            "root_rot",
            "fps",
        ]

        for key in required:
            if key not in d:
                raise KeyError(
                    f"{src} missing field: {key}"
                )

        converted = {
            "dof_pos": d["dof"],
            "root_pos": d["root_trans_offset"],

            # 保持 GRAIL 原始 xyzw。
            # GMR_to_instinct.py 会负责 xyzw -> wxyz。
            "root_rot": d["root_rot"],

            "fps": d["fps"],
        }

        filename = os.path.basename(repo_path)

        dst = os.path.join(
            output_dir,
            filename,
        )

        with open(dst, "wb") as f:
            pickle.dump(
                converted,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        total += 1

        if i % 50 == 0 or i == len(repo_files):
            print(
                f"{category}: "
                f"{i}/{len(repo_files)}"
            )


print()
print("=" * 60)
print("DONE")
print("Total converted:", total)
print("=" * 60)
