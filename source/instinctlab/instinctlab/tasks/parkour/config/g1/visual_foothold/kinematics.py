"""Torch leg FK, evaluated from the project's URDF without simulator state."""

import math
import xml.etree.ElementTree as ET

import torch


def rotate(rotation, vector):
    """Apply column-vector rotation with arbitrary leading batch dimensions."""
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


def axis_rotation(axis, angle):
    """Rodrigues rotation for a unit axis and a batch of angles."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(3, 3)
    return torch.eye(3, device=angle.device) + angle.sin()[..., None, None] * skew + (
        1 - angle.cos()[..., None, None]
    ) * (skew @ skew)


def quaternion_rotation(q):
    """Convert wxyz quaternions to rotation matrices."""
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)).unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


class LegKinematics:
    """Include fixed transforms, waist ancestors, and the nominal shoe offset."""

    def __init__(self, path, joint_names, device):
        tree = ET.parse(path).getroot()
        joints = {j.find("child").get("link"): j for j in tree.findall("joint")}
        links = {link.get("name") for link in tree.findall("link")}
        roots = links - set(joints)
        if roots != {"torso_link"} or len(set(joint_names)) != len(joint_names):
            raise ValueError("Visual foothold FK requires a unique torso_link root and joint names")
        self.chains = []
        self.offset = torch.tensor((0.039, 0., -0.058), device=device)
        for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
            chain, current = [], foot
            while current != "torso_link":
                joint = joints[current]
                kind = joint.get("type")
                if kind not in ("fixed", "revolute", "continuous"):
                    raise ValueError(f"Unsupported leg joint: {kind}")
                origin = joint.find("origin")
                xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
                rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
                r = torch.eye(3, device=device)
                for axis, angle in zip(((0., 0., 1.), (0., 1., 0.), (1., 0., 0.)), reversed(rpy)):
                    r = r @ axis_rotation(torch.tensor(axis, device=device), torch.tensor(angle, device=device))
                axis_node = joint.find("axis")
                axis = torch.tensor([float(v) for v in (
                    axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0"
                ).split()], device=device)
                axis = axis / axis.norm().clamp_min(1e-9)
                index = None if kind == "fixed" else joint_names.index(joint.get("name"))
                chain.insert(0, (index, torch.tensor(xyz, device=device), r, axis))
                current = joint.find("parent").get("link")
            self.chains.append(chain)

    def compute(self, q, dq):
        """Return sole position, orientation and body-relative position derivative."""
        positions, rotations, velocities = [], [], []
        for chain in self.chains:
            p = torch.zeros(q.shape[0], 3, device=q.device)
            r = torch.eye(3, device=q.device).expand(q.shape[0], 3, 3).clone()
            axes = []
            for index, xyz, origin_r, axis in chain:
                p = p + rotate(r, xyz.expand_as(p))
                r = r @ origin_r
                if index is not None:
                    axes.append((index, rotate(r, axis.expand_as(p)), p.clone()))
                    r = r @ axis_rotation(axis, q[:, index])
            p = p + rotate(r, self.offset.expand_as(p))
            v = torch.zeros_like(p)
            for index, axis, joint_p in axes:
                v += torch.cross(axis, p - joint_p, dim=-1) * dq[:, index, None]
            positions.append(p)
            rotations.append(r)
            velocities.append(v)
        return torch.stack(positions, 1), torch.stack(rotations, 1), torch.stack(velocities, 1)


class ContactObserver:
    """Hysteretic contact FSM with first-candidate events and per-env reset."""

    def __init__(self, n, dt, device, on=20., off=10., confirm_s=0.02, cutoff_hz=10.):
        if not 0 <= off < on or not 0 < cutoff_hz < 0.5 / dt:
            raise ValueError("Invalid observer thresholds/cutoff")
        self.dt, self.on, self.off, self.confirm = dt, on, off, confirm_s
        self.alpha = 1 - math.exp(-2 * math.pi * cutoff_hz * dt)
        self.state = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.timer = torch.zeros(n, 2, device=device)
        self.armed = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)
        self.velocity = torch.zeros(n, 2, 3, device=device)

    def reset(self, ids):
        self.state[ids] = 0
        self.timer[ids] = 0
        self.armed[ids] = False
        self.velocity[ids] = 0
        self.fresh[ids] = True

    def update(self, velocity, force, advance):
        fresh = self.fresh.clone()
        active = advance[:, None]
        self.velocity = torch.where(fresh[:, None, None], velocity, torch.where(
            active[..., None], self.velocity + self.alpha * (velocity - self.velocity), self.velocity
        ))
        high, low = force >= self.on, force <= self.off
        old = self.state.clone()
        first = fresh[:, None]
        # Initial standing is support, never a fictitious touchdown.
        self.state = torch.where(first & high, 2, self.state)
        self.armed |= active & (old == 0) & low
        candidate = active & ~first & (old == 0) & high
        self.state[candidate] = 1
        self.timer[candidate] = 0
        continued = active & (old == 1) & high
        self.timer[continued] += self.dt
        canceled = active & (old == 1) & ~high
        self.state[canceled] = 0
        confirmed = active & (self.state == 1) & high & (self.timer + 1e-6 >= self.confirm)
        touchdown = confirmed & self.armed
        self.state[confirmed] = 2
        self.armed[confirmed] = False
        release = active & (old == 2) & low
        self.state[release] = 3
        self.timer[release] = 0
        continued = active & (old == 3) & low
        self.timer[continued] += self.dt
        self.state[active & (old == 3) & ~low] = 2
        airborne = active & (self.state == 3) & low & (self.timer + 1e-6 >= self.confirm)
        self.state[airborne] = 0
        self.armed[airborne] = True
        self.fresh[:] = False
        return candidate, touchdown
