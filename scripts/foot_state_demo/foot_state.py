"""Small, batched foot kinematics and contact observer; no simulator imports.

Positions and position derivatives are expressed relative to the URDF root.
Contact scores describe a force heuristic, not calibrated probabilities.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
STATE_NAMES = ("air", "contact_candidate", "support", "release_candidate")
AIR, CANDIDATE, SUPPORT, RELEASE = range(4)


def rotation(axis, angle):
    """Rodrigues rotation, with an arbitrary leading batch dimension."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.asarray(angle)[..., None, None]
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def quaternion_matrix(quaternion):
    """Convert project-convention wxyz quaternions to rotation matrices."""
    q = np.asarray(quaternion, dtype=float)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or (norms < 1e-12).any():
        raise ValueError("Quaternion must be finite and nonzero")
    w, x, y, z = np.moveaxis(q / norms, -1, 0)
    return np.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3))


def relative_positions(position, orientation):
    """Return each foot in the OTHER foot's current sole frame, shape [N, 2, 3]."""
    delta = position - position[:, ::-1]
    return np.einsum("nfji,nfj->nfi", orientation[:, ::-1], delta)


class LegKinematics:
    """Evaluate only root-to-foot ancestors of the actual URDF.

    Uses the standard URDF origin * joint-motion convention. Fixed joints and
    waist ancestors are included; no dynamics, meshes, or learned model needed.
    The nominal sole is a named coordinate convention, not a contact centroid.
    """

    def __init__(self, urdf, joint_names, foot_links=FOOT_LINKS, sole_offset=(0.039, 0, -0.058)):
        self.urdf = Path(urdf)
        tree = ET.parse(self.urdf).getroot()
        links = {link.attrib["name"] for link in tree.findall("link")}
        by_child = {joint.find("child").attrib["link"]: joint for joint in tree.findall("joint")}
        roots = links - set(by_child)
        if len(roots) != 1:
            raise ValueError("Expected one URDF root")
        self.root_link = roots.pop()
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Joint names must be unique")
        self.joint_names = tuple(joint_names)
        self.foot_links = tuple(foot_links)
        if len(self.foot_links) != 2:
            raise ValueError("Exactly two foot links are required")
        self.sole_offset = np.asarray(sole_offset, dtype=float)
        if self.sole_offset.shape != (3,) or not np.isfinite(self.sole_offset).all():
            raise ValueError("sole_offset must contain three finite metres")
        self.chains = []
        used = set()
        for foot in self.foot_links:
            if foot not in links:
                raise ValueError(f"Missing foot link: {foot}")
            chain, current, visited = [], foot, set()
            while current != self.root_link:
                if current in visited or current not in by_child:
                    raise ValueError("Disconnected/cyclic URDF chain")
                visited.add(current)
                joint = by_child[current]
                name, kind = joint.attrib["name"], joint.attrib["type"]
                if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                    raise ValueError(f"Unsupported joint type: {kind}")
                origin = joint.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                axis_node = joint.find("axis")
                axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
                if xyz.shape != (3,) or rpy.shape != (3,) or axis.shape != (3,):
                    raise ValueError(f"Malformed joint geometry: {name}")
                if not np.isfinite(np.concatenate((xyz, rpy, axis))).all() or np.linalg.norm(axis) < 1e-12:
                    raise ValueError(f"Invalid joint geometry: {name}")
                origin_rotation = (
                    rotation((0, 0, 1), rpy[2]) @ rotation((0, 1, 0), rpy[1]) @ rotation((1, 0, 0), rpy[0])
                )
                index = None if kind == "fixed" else self.joint_names.index(name)
                if index is not None:
                    used.add(name)
                chain.insert(0, (kind, index, xyz, origin_rotation, axis / np.linalg.norm(axis)))
                current = joint.find("parent").attrib["link"]
            self.chains.append(chain)
        self.required_joint_names = tuple(name for name in self.joint_names if name in used)

    def compute(self, joint_position, joint_velocity):
        """Return sole positions, rotations, and relative position derivatives."""
        q, dq = np.asarray(joint_position, dtype=float), np.asarray(joint_velocity, dtype=float)
        if q.ndim != 2 or q.shape != dq.shape or q.shape[1] != len(self.joint_names):
            raise ValueError("Expected q and dq shaped [environments, joint_names]")
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError("Nonfinite joint measurement")
        positions, orientations, velocities = [], [], []
        for chain in self.chains:
            p = np.zeros((len(q), 3))
            r = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
            axes = []
            for kind, index, origin, origin_rotation, axis in chain:
                p += np.einsum("nij,j->ni", r, origin)
                r = r @ origin_rotation
                if index is None:
                    continue
                axis_b = np.einsum("nij,j->ni", r, axis)
                axes.append((kind, index, axis_b, p.copy()))
                if kind == "prismatic":
                    p += axis_b * q[:, index, None]
                else:
                    r = r @ rotation(axis, q[:, index])
            p += np.einsum("nij,j->ni", r, self.sole_offset)
            v = np.zeros_like(p)
            for kind, index, axis_b, joint_origin in axes:
                column = axis_b if kind == "prismatic" else np.cross(axis_b, p - joint_origin)
                v += column * dq[:, index, None]
            positions.append(p)
            orientations.append(r)
            velocities.append(v)
        return np.stack(positions, axis=1), np.stack(orientations, axis=1), np.stack(velocities, axis=1)


@dataclass(frozen=True)
class ObserverConfig:
    dt: float = 0.02
    velocity_cutoff_hz: float = 10.0
    force_on_n: float = 20.0
    force_off_n: float = 10.0
    confirm_on_s: float = 0.02
    confirm_off_s: float = 0.02

    def __post_init__(self):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Observer parameters must be finite")
        if self.dt <= 0 or not 0 < self.velocity_cutoff_hz < 0.5 / self.dt:
            raise ValueError("Require dt > 0 and cutoff strictly below Nyquist")
        if not 0 <= self.force_off_n < self.force_on_n or min(self.confirm_on_s, self.confirm_off_s) < 0:
            raise ValueError("Require 0 <= off < on force and nonnegative confirmation times")


class FootStateObserver:
    """Per-environment contact FSM and velocity low-pass with reset isolation.

    A repeated step_id returns the cached result. Forces must be the upward
    component in newtons, NOT a net-force norm. This is a support proxy and
    does not establish full-sole contact, no slip, or stable balance.
    """

    def __init__(self, num_envs, config=ObserverConfig()):
        if not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.config = config
        self.num_envs = num_envs
        shape = (num_envs, 2)
        self.state = np.zeros(shape, dtype=int)
        self.timer = np.zeros(shape)
        self.initialized = np.zeros(shape, dtype=bool)
        self.armed = np.zeros(shape, dtype=bool)
        self.filtered_velocity = np.zeros(shape + (3,))
        self.candidate_time = np.zeros(shape)
        self.release_time = np.zeros(shape)
        self.candidate_relative_position = np.zeros(shape + (3,))
        self.candidate_reference_valid = np.zeros(shape, dtype=bool)
        self.reference_foot = np.full(num_envs, -1, dtype=int)
        self.last_step_id = None
        self.cached = None

    def reset(self, env_ids):
        ids = np.asarray(env_ids, dtype=int)
        for array in (self.state, self.timer, self.initialized, self.armed, self.filtered_velocity,
                      self.candidate_time, self.release_time, self.candidate_relative_position,
                      self.candidate_reference_valid):
            array[ids] = 0
        self.reference_foot[ids] = -1
        # Keep the last step's immutable result for readers after auto-reset.

    def update(self, step_id, time_s, position, orientation, velocity, upward_force):
        if step_id == self.last_step_id:
            return self.cached
        if self.last_step_id is not None and step_id < self.last_step_id:
            raise ValueError("step_id must increase monotonically")
        p, r, v, force = (np.asarray(value, dtype=float) for value in (position, orientation, velocity, upward_force))
        shape = (self.num_envs, 2)
        if p.shape != shape + (3,) or v.shape != p.shape or r.shape != shape + (3, 3) or force.shape != shape:
            raise ValueError("Unexpected foot-state batch shape")
        if not math.isfinite(time_s) or not all(np.isfinite(value).all() for value in (p, r, v, force)):
            raise ValueError("Nonfinite foot observation")
        c = self.config
        relative = relative_positions(p, r)
        first = ~self.initialized
        self.filtered_velocity[first] = v[first]
        alpha = 1 - math.exp(-2 * math.pi * c.velocity_cutoff_hz * c.dt)
        self.filtered_velocity[~first] += alpha * (v[~first] - self.filtered_velocity[~first])
        self.initialized[:] = True
        high, low = force >= c.force_on_n, force <= c.force_off_n
        old = self.state.copy()
        self.armed |= (old == AIR) & low
        start_on = (old == AIR) & high
        self.state[start_on] = CANDIDATE
        self.timer[start_on] = 0
        self.candidate_time[start_on] = time_s
        self.candidate_relative_position[start_on] = relative[start_on]
        continuing_on = (old == CANDIDATE) & high
        self.timer[continuing_on] += c.dt
        cancel_on = (old == CANDIDATE) & ~high
        self.state[cancel_on] = AIR
        self.timer[cancel_on] = 0
        confirmed_on = (self.state == CANDIDATE) & high & (self.timer + 1e-10 >= c.confirm_on_s)
        touchdown = confirmed_on & self.armed
        self.state[confirmed_on] = SUPPORT
        self.armed[confirmed_on] = False
        self.timer[confirmed_on] = 0
        start_off = (old == SUPPORT) & low
        self.state[start_off] = RELEASE
        self.timer[start_off] = 0
        self.release_time[start_off] = time_s
        continuing_off = (old == RELEASE) & low
        self.timer[continuing_off] += c.dt
        cancel_off = (old == RELEASE) & ~low
        self.state[cancel_off] = SUPPORT
        self.timer[cancel_off] = 0
        liftoff = (self.state == RELEASE) & low & (self.timer + 1e-10 >= c.confirm_off_s)
        self.state[liftoff] = AIR
        self.armed[liftoff] = True
        self.timer[liftoff] = 0
        valid = self.state == SUPPORT
        self.candidate_reference_valid[start_on] = (valid[:, ::-1] & (force[:, ::-1] > c.force_off_n))[start_on]
        for env_id in range(self.num_envs):
            previous = self.reference_foot[env_id]
            if previous < 0 or not valid[env_id, previous]:
                choices = np.flatnonzero(valid[env_id])
                self.reference_foot[env_id] = choices[0] if len(choices) else -1
        result = {
            "position_b": p.copy(), "orientation_b": r.copy(),
            "velocity_b": self.filtered_velocity.copy(), "relative_to_other": relative,
            "state": self.state.copy(), "support_valid": valid.copy(),
            "reference_foot": self.reference_foot.copy(),
            "support_count": valid.sum(axis=1), "touchdown": touchdown, "liftoff": liftoff,
            "candidate_time_s": self.candidate_time.copy(),
            "release_candidate_time_s": self.release_time.copy(),
            "candidate_relative_position": self.candidate_relative_position.copy(),
            "candidate_reference_valid": self.candidate_reference_valid.copy(),
            "contact_score": np.clip((force - c.force_off_n) / (c.force_on_n - c.force_off_n), 0, 1),
        }
        self.last_step_id, self.cached = step_id, result
        return result
