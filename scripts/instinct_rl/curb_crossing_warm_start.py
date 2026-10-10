"""Load compatible policy/value weights without old AMP or optimizer state."""

import hashlib
import torch
import yaml
from pathlib import Path


def warm_start(runner, checkpoint, agent_cfg):
    """Initialize a new experiment; strict shapes and observation recipe are required."""
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    source = yaml.safe_load((checkpoint.parent / "params/agent.yaml").read_text(encoding="utf-8"))
    target = agent_cfg.to_dict()
    if source.get("policy") != target.get("policy"):
        raise ValueError("Warm-start requires the same policy architecture/settings as the source run")
    if source.get("empirical_normalization") or target.get("empirical_normalization"):
        raise ValueError("Legacy empirical normalization is not supported for curb warm-start")
    if source.get("normalizers", {}) != target.get("normalizers", {}):
        raise ValueError("Warm-start normalizer configuration differs from source")
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    runner.alg.actor_critic.load_state_dict(saved["model_state_dict"], strict=True)
    for name, normalizer in runner.normalizers.items():
        normalizer.load_state_dict(saved[name + "_normalizer_state_dict"], strict=True)
    # Do not call runner.load(): discriminator, optimizers, counters and replay
    # buffers belong to this new reference distribution and start fresh.
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(
        checkpoint=str(checkpoint),
        sha256=digest.hexdigest(),
        source_iteration=saved.get("iter"),
        scope="policy/value/normalizer weights only; fresh AMP, optimizers and iteration counter",
    )
