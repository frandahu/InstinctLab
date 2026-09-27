import os
import random
import time

from huggingface_hub import HfApi, hf_hub_download


REPO = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"

LOCAL_DIR = "/workspace/instinctlab/data/grail_raw"

TARGETS = {
    "curb": 500,
    "slope": 500,
    "stair_p1": 250,
    "stair_p2": 250,
}

SEED = 42
MAX_RETRIES = 10

api = HfApi()

manifest_dir = os.path.join(LOCAL_DIR, "manifests")
os.makedirs(manifest_dir, exist_ok=True)

all_failed = []


for category_id, (category, count) in enumerate(TARGETS.items()):

    print()
    print("=" * 70)
    print(f"Processing {category}")
    print("=" * 70)

    manifest_path = os.path.join(
        manifest_dir,
        f"{category}.txt"
    )

    # --------------------------------------------------
    # 如果 manifest 已存在，就继续使用之前选中的文件
    # 不重新随机抽样
    # --------------------------------------------------
    if os.path.isfile(manifest_path):

        with open(manifest_path, "r") as f:
            selected = [
                line.strip()
                for line in f
                if line.strip()
            ]

        print(
            f"Using existing manifest: "
            f"{manifest_path}"
        )

        print(
            f"Selected files: {len(selected)}"
        )

    else:

        repo_dir = f"data/{category}/robot"

        print(f"Scanning {repo_dir}")

        items = api.list_repo_tree(
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=repo_dir,
            recursive=False,
        )

        files = sorted(
            item.path
            for item in items
            if item.path.endswith(".pkl")
        )

        if len(files) < count:
            raise RuntimeError(
                f"{category}: need {count}, "
                f"but only found {len(files)}"
            )

        rng = random.Random(
            SEED + category_id
        )

        selected = sorted(
            rng.sample(files, count)
        )

        with open(manifest_path, "w") as f:
            for filename in selected:
                f.write(filename + "\n")

        print(
            f"Created manifest: "
            f"{manifest_path}"
        )

    category_failed = []

    # --------------------------------------------------
    # 下载
    # --------------------------------------------------
    for i, filename in enumerate(selected, 1):

        local_path = os.path.join(
            LOCAL_DIR,
            filename
        )

        # 已经完整存在的文件直接跳过
        # 不再访问 Hugging Face
        if (
            os.path.isfile(local_path)
            and os.path.getsize(local_path) > 0
        ):

            print(
                f"[{category}] "
                f"{i:04d}/{len(selected):04d} "
                f"SKIP "
                f"{os.path.basename(filename)}"
            )

            continue

        success = False

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                print(
                    f"[{category}] "
                    f"{i:04d}/{len(selected):04d} "
                    f"DOWNLOAD "
                    f"{os.path.basename(filename)} "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )

                hf_hub_download(
                    repo_id=REPO,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=LOCAL_DIR,
                )

                success = True
                break

            except Exception as e:

                print()
                print(
                    f"Download failed:"
                )
                print(filename)

                print(
                    f"{type(e).__name__}: {e}"
                )

                if attempt < MAX_RETRIES:

                    wait = min(
                        60,
                        2 ** attempt
                    )

                    print(
                        f"Retrying in {wait}s..."
                    )

                    time.sleep(wait)

        if not success:

            category_failed.append(
                filename
            )

            all_failed.append(
                filename
            )

            print(
                f"FAILED after "
                f"{MAX_RETRIES} attempts:"
            )

            print(filename)

            # 不退出
            # 继续下载下一条


    print()
    print(
        f"{category}: "
        f"{len(selected) - len(category_failed)} "
        f"available, "
        f"{len(category_failed)} failed"
    )


print()
print("=" * 70)
print("DOWNLOAD PASS COMPLETE")
print("=" * 70)

if all_failed:

    failed_file = os.path.join(
        manifest_dir,
        "failed_downloads.txt"
    )

    with open(failed_file, "w") as f:
        for filename in all_failed:
            f.write(filename + "\n")

    print(
        f"Still failed: "
        f"{len(all_failed)}"
    )

    print(
        f"Saved to: {failed_file}"
    )

else:

    failed_file = os.path.join(
        manifest_dir,
        "failed_downloads.txt"
    )

    if os.path.isfile(failed_file):
        os.remove(failed_file)

    print(
        "All selected files are available."
    )
