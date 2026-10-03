"""Capture the actual server RL implementation alongside a new stair run."""

import hashlib
import inspect
import json
from pathlib import Path


def save_runtime_sources(runner, env_cfg, log_dir):
    output = Path(log_dir) / "params"
    output.mkdir(parents=True, exist_ok=True)
    algorithm = runner.alg
    actor = algorithm.actor_critic
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    metadata = dict(
        step_dt=step_dt, amp_reward_rate=env_cfg.amp_reward_rate,
        discriminator_reward_coef=algorithm.discriminator_reward_coef,
        algorithm_class=type(algorithm).__module__ + ":" + type(algorithm).__name__,
        policy_class=type(actor).__module__ + ":" + type(actor).__name__,
        min_noise_std=actor.min_noise_std, max_noise_std=actor.max_noise_std,
        implementations={},
    )
    # Include each implementation in the MRO: Wasabi delegates reward addition
    # and PPO updates to its parent. Never assume the installed fork matches main.
    for cls in type(algorithm).__mro__:
        for name in ("process_env_step", "compute_auxiliary_reward", "update"):
            if name not in cls.__dict__:
                continue
            function = cls.__dict__[name]
            key = cls.__name__ + "_" + name
            try:
                source = inspect.getsource(function)
                path = inspect.getsourcefile(function)
            except (OSError, TypeError):
                metadata["implementations"][key] = {"source_available": False}
                continue
            (output / (key + ".txt")).write_text(source, encoding="utf-8")
            metadata["implementations"][key] = dict(
                source_available=True, path=path,
                sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )
    (output / "stair_runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[STAIRS] Actual RL source and reward/noise settings saved in {output}", flush=True)
