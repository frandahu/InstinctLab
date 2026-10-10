"""Task-local curb contacts and episode state, using the existing ankle sensors.

Privileged geometry is used for training rewards only. Policy observations stay
unchanged. Contact plus a nominal sole envelope is not a no-slip certificate.
"""

import math
import torch


class CurbCrossingState:
    """Count each forward platform transition once, after sustained support.

    A ground touch permanently invalidates that short crossing for this episode.
    Reward/termination calls share one snapshot per physics control step. Reset
    only clears the specified environments, including contact confirmation.
    """

    def __init__(self, env):
        self.env = env
        terrain = env.scene.terrain
        cases = [case for row in terrain.terrain_generator.grid_cases for case in row]
        self.grid = {
            key: torch.tensor(value, dtype=torch.float32, device=env.device)
            for key, value in {
                "starts": [[c["start_x_m"] for c in case["curbs"]] for case in cases],
                "ends": [[c["end_x_m"] for c in case["curbs"]] for case in cases],
                "heights": [[c["height_m"] for c in case["curbs"]] for case in cases],
                "required": [case["bridge_required"] for case in cases],
                "goal": [case["goal_x_m"] for case in cases],
                "end": [case["stair_end_x_m"] for case in cases],
                "width": [case["width_m"] for case in cases],
            }.items()
        }
        self.feet, self.contacts = [], []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            for asset, ids in ((env.scene["robot"], self.feet), (env.scene["contact_forces"], self.contacts)):
                found, _ = asset.find_bodies(name)
                if len(found) != 1:
                    raise ValueError("Curb task requires one body named " + name)
                ids.append(found[0])
        n = env.num_envs
        self.last_platform = torch.full((n,), -1, dtype=torch.long, device=env.device)
        self.ground_seen = torch.zeros(n, 4, dtype=torch.bool, device=env.device)
        self.passed = torch.zeros_like(self.ground_seen)
        self.support_platform = torch.full((n, 2), -1, dtype=torch.long, device=env.device)
        self.support_time = torch.zeros(n, 2, device=env.device)
        self.finish_time = torch.zeros(n, device=env.device)
        self.bridge_event = torch.zeros(n, device=env.device)
        self.ground_contact = torch.zeros(n, device=env.device)
        self.finished = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.step = -1

    def reset(self, env_ids):
        self.last_platform[env_ids] = -1
        self.support_platform[env_ids] = -1
        for tensor in (
            self.ground_seen,
            self.passed,
            self.support_time,
            self.finish_time,
            self.bridge_event,
            self.ground_contact,
            self.finished,
        ):
            tensor[env_ids] = 0
        # Keep other environments' current-step result cached through auto-reset.

    def update(self):
        env = self.env
        if self.step == env.common_step_counter:
            return self
        self.step = env.common_step_counter
        terrain = env.scene.terrain
        index = terrain.terrain_levels * terrain.cfg.terrain_generator.num_cols + terrain.terrain_types
        data = {key: value[index] for key, value in self.grid.items()}
        robot = env.scene["robot"].data
        q = robot.body_quat_w[:, self.feet]
        # wxyz quaternion rotation, including the shoe's forward sole offset.
        offset = torch.tensor((0.039, 0.0, -0.058), device=env.device).expand_as(q[..., 1:])
        cross = 2 * torch.cross(q[..., 1:], offset, dim=-1)
        sole = robot.body_pos_w[:, self.feet] + offset + q[..., :1] * cross + torch.cross(q[..., 1:], cross, dim=-1)
        sole = sole - env.scene.env_origins[:, None, :]
        x, y, z = sole.unbind(dim=-1)
        w, qx, qy, qz = q.unbind(dim=-1)
        # Project the rectangular sole onto lane axes, accounting for foot rotation.
        xx, xy = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz)
        yx, yy = 2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz)
        extent_x = 0.093 * xx.abs() + 0.036 * xy.abs()
        extent_y = 0.093 * yx.abs() + 0.036 * yy.abs()
        contact = env.scene["contact_forces"].data.net_forces_w[:, self.contacts, 2] > 20.0
        on_platform = (
            (x[..., None] - extent_x[..., None] >= data["starts"][:, None] + 0.025)
            & (x[..., None] + extent_x[..., None] <= data["ends"][:, None] - 0.025)
            & ((z[..., None] - data["heights"][:, None]).abs() < 0.04)
            & (y.abs() + extent_y < data["width"][:, None] / 2 - 0.025)[..., None]
            & contact[..., None]
        )
        platform = torch.where(on_platform.any(dim=-1), on_platform.long().argmax(dim=-1), -1)
        same = (platform == self.support_platform) & (platform >= 0)
        self.support_time = torch.where(
            same, self.support_time + env.step_dt, torch.where(platform >= 0, env.step_dt, 0.0)
        )
        self.support_platform[:] = platform
        stable = torch.where(self.support_time >= 0.06 - 1e-6, platform, -1).max(dim=-1).values
        in_gap = (x[..., None] >= data["ends"][:, None, :-1]) & (x[..., None] < data["starts"][:, None, 1:])
        low_contact = in_gap & (z.abs() < 0.04)[..., None] & contact[..., None]
        self.ground_seen |= low_contact.any(dim=1)
        required = data["required"].bool()
        self.ground_contact[:] = (low_contact & required[:, None]).any(dim=-1).float().sum(dim=-1)
        self.bridge_event.zero_()
        upright = robot.projected_gravity_b[:, 2] < -math.cos(0.5)
        for gap in range(4):
            arrived = (self.last_platform == gap) & (stable == gap + 1) & ~self.passed[:, gap]
            clean = arrived & ~self.ground_seen[:, gap] & upright
            # The long gap retains ground walking. A short gap must stay clean.
            accepted = torch.where(required[:, gap], clean, arrived & self.ground_seen[:, gap])
            self.passed[:, gap] |= accepted
            self.bridge_event += (clean & required[:, gap]).float()
        self.last_platform[:] = torch.maximum(self.last_platform, stable)
        root = robot.root_pos_w - env.scene.env_origins
        final_support = ((x > data["end"][:, None] + 0.1) & (z.abs() < 0.04) & contact).all(dim=-1)
        complete = (self.passed | ~required).all(dim=-1) & ~(self.ground_seen & required).any(dim=-1)
        all_platforms = (self.last_platform == 4) | ~required.any(dim=-1)
        at_goal = (
            (root[:, 0] > data["goal"] - 0.30)
            & (root[:, 1].abs() < 0.35)
            & final_support
            & upright
            & (robot.root_lin_vel_b[:, :2].norm(dim=-1) < 0.5)
            & complete
            & all_platforms
        )
        self.finish_time = torch.where(at_goal, self.finish_time + env.step_dt, 0.0)
        self.finished[:] = self.finish_time >= 0.5 - 1e-6
        return self


def crossing_state(env):
    if not hasattr(env, "_curb_crossing_state"):
        env._curb_crossing_state = CurbCrossingState(env)
    return env._curb_crossing_state.update()


def reset_curb_crossing(env, env_ids):
    if hasattr(env, "_curb_crossing_state"):
        state = env._curb_crossing_state
        required = state.grid["required"][
            env.scene.terrain.terrain_levels * env.scene.terrain.cfg.terrain_generator.num_cols
            + env.scene.terrain.terrain_types
        ].bool()
        # Curriculum may already have changed rows. All non-flat rows have the
        # same required-gap mask, so episode counts remain comparable.
        # Isaac Lab clears extras['log'] AFTER reset events. CommandManager.reset
        # aggregates these metrics later in the same reset, before clearing them.
        metrics = env.command_manager.get_term("base_velocity").metrics
        metrics["curb_clean_crossings"][env_ids] = (state.passed[env_ids] & required[env_ids]).float().sum(-1)
        metrics["curb_ground_touched_short_gaps"][env_ids] = (
            (state.ground_seen[env_ids] & required[env_ids]).float().sum(-1)
        )
        metrics["curb_route_success"][env_ids] = state.finished[env_ids].float()
        state.reset(env_ids)


def curb_bridge_reward(env):
    """One event bonus per clean crossing; cancel the manager's dt scaling."""
    return crossing_state(env).bridge_event / env.step_dt


def curb_gap_ground_contact(env):
    """Per-foot contact duration cost, only inside required short gaps."""
    return crossing_state(env).ground_contact


def curb_crossing_goal(env):
    return crossing_state(env).finished


def curb_crossing_curriculum(env, env_ids):
    """Advance only after the entire clean route and the stable final support."""
    terrain = env.scene.terrain
    success = env.termination_manager.get_term("curb_goal")[env_ids]
    failed = torch.zeros_like(success)
    for name in ("root_height", "off_course", "bad_orientation", "base_contact", "nonfinite_state"):
        failed |= env.termination_manager.get_term(name)[env_ids]
    promote = success & ~failed
    demote = (env.episode_length_buf[env_ids] > 0) & ~promote
    promote &= terrain.terrain_levels[env_ids] < terrain.cfg.terrain_generator.num_rows - 1
    terrain.update_env_origins(env_ids, promote, demote)
    return terrain.terrain_levels.float().mean()
