"""Demo-only sampler at the existing pre-reset success-term boundary.

Import only after AppLauncher starts. The inherited success decision and reset
behavior are unchanged. No module or class is patched globally.
"""

from instinctlab.tasks.parkour.config.g1.stair_eval_cfg import StairSuccess


class FootSnapshotSuccess(StairSuccess):
    """Keep terminal joint/contact measurements intact across automatic resets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids, _ = env.scene["robot"].find_bodies(["torso_link"], preserve_order=True)
        if len(ids) != 1:
            raise RuntimeError("Foot demo requires the torso_link URDF root")
        self.torso_id = ids[0]
        self.foot_snapshot = None

    def __call__(self, env, hold_s=0.5, goal_radius=0.35):
        success = super().__call__(env, hold_s=hold_s, goal_radius=goal_radius)
        data = env.scene["robot"].data
        self.foot_snapshot = {
            "q": data.joint_pos.clone(), "dq": data.joint_vel.clone(),
            "root_position_w": data.body_pos_w[:, self.torso_id].clone(),
            "root_quaternion_w": data.body_quat_w[:, self.torso_id].clone(),
            "ankle_position_w": data.body_pos_w[:, self.foot_ids].clone(),
            "ankle_quaternion_w": data.body_quat_w[:, self.foot_ids].clone(),
            "force_w": env.scene["contact_forces"].data.net_forces_w[:, self.contact_ids].clone(),
            "episode_step": env.episode_length_buf.clone(),
        }
        return success
