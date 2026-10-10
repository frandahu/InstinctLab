"""Exercise the actual URDF/FSM on synthetic measurements, WITHOUT Isaac Sim.

This does not simulate locomotion or validate a trained checkpoint. It writes
CSV, JSON, and a PNG panel; MP4 of the robot is provided by run_demo.py.
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np

from demo_output import DemoOutput, overlay_frame
from foot_state import FootStateObserver, LegKinematics, ObserverConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = ROOT / "source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/foot_state_demo/offline")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.num_envs) <= 0:
        parser.error("steps and num_envs must be positive")
    sys.path.insert(0, str(ROOT / "scripts/instinct_rl"))
    from eval_stairs import create_output_directory

    output = create_output_directory(args.output_dir)
    joints = [node.attrib["name"] for node in ET.parse(args.urdf).getroot().findall("joint")
              if node.attrib["type"] != "fixed"]
    fk = LegKinematics(args.urdf, joints)
    config = ObserverConfig()
    observer = FootStateObserver(args.num_envs, config)
    log = DemoOutput(output)
    episodes = np.zeros(args.num_envs, dtype=int)
    episode_steps = np.zeros(args.num_envs, dtype=int)
    report = {"mode": "synthetic_measurements_no_physics", "status": "running",
              "observer_config": asdict(config), "urdf": str(args.urdf), "root_link": fk.root_link}
    state = None
    try:
        for step in range(1, args.steps + 1):
            t = step * config.dt
            phase = 2 * np.pi * t + np.arange(args.num_envs)[:, None] * 0.3 + np.arange(len(joints))[None] * 0.2
            q, dq = 0.10 * np.sin(phase), 0.10 * 2 * np.pi * np.cos(phase)
            force = np.zeros((args.num_envs, 2, 3))
            cycle = (t + np.arange(args.num_envs) * 0.1) % 1.0
            force[:, 0, 2] = np.where(cycle < 0.55, 150, 0)
            force[:, 1, 2] = np.where((cycle >= 0.45) | (cycle < 0.05), 150, 0)
            # A lateral strike alone must not become upward support evidence.
            force[(cycle > 0.70) & (cycle < 0.76), 0, 0] = 200
            started = time.perf_counter()
            p, r, v = fk.compute(q, dq)
            state = observer.update(step, t, p, r, v, force[..., 2])
            log.update_ms.append((time.perf_counter() - started) * 1000)
            episode_steps += 1
            reset = np.zeros(args.num_envs, dtype=bool)
            if step == args.steps // 2 and args.num_envs > 1:
                reset[1] = True
            log.append(step, t, state, force, episodes, episode_steps, reset,
                       ["synthetic_reset" if value else "" for value in reset], np.ones(args.num_envs, dtype=bool))
            observer.reset(np.flatnonzero(reset))
            episodes[reset] += 1
            episode_steps[reset] = 0
        from PIL import Image

        frame = overlay_frame(np.full((360, 640, 3), 40, dtype=np.uint8), state, 0, args.steps)
        Image.fromarray(frame).save(output / "offline_preview.png")
        report["status"] = "synthetic_complete"
    except BaseException as error:
        report.update({"status": "error", "error": str(error)})
        raise
    finally:
        log.close(report)
    print(f"[OFFLINE ONLY] Synthetic measurements; no policy or physics: {output}")


if __name__ == "__main__":
    main()
