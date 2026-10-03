"""Per-step inference diagnostics; importing this module does not start Isaac Sim."""

DIAGNOSTIC_COLUMNS = [
    "policy_action_rms", "policy_action_delta_rms",
    "observed_command_x_min_m_s", "observed_command_x_max_m_s",
    "depth_input_min", "depth_input_max", "depth_input_mean",
    "joint_velocity_rms_rad_s", "applied_torque_rms_nm",
]


def policy_step_diagnostics(observation, actions, previous_actions, obs_format):
    """Inspect the actual flattened actor input, before normalizing or stepping.

    Command extrema cover all history frames, so no history ordering is assumed.
    Depth statistics are of the trained, processed camera input, not raw metres.
    """
    import torch

    offsets, offset = {}, 0
    for name, shape in obs_format.items():
        size = 1
        for dimension in shape:
            size *= int(dimension)
        offsets[name] = (offset, offset + size)
        offset += size
    if observation.shape[1] != offset:
        raise ValueError("Policy observation width does not match the configured segments")
    command_start, command_end = offsets["velocity_commands"]
    command = observation[:, command_start:command_end].reshape(observation.shape[0], -1, 3)
    depth_start, depth_end = offsets["depth_image"]
    depth = observation[:, depth_start:depth_end]
    difference = torch.zeros_like(actions) if previous_actions is None else actions - previous_actions
    return {
        "policy_action_rms": actions.square().mean(dim=-1).sqrt(),
        "policy_action_delta_rms": difference.square().mean(dim=-1).sqrt(),
        "observed_command_x_min_m_s": command[:, :, 0].min(dim=-1).values,
        "observed_command_x_max_m_s": command[:, :, 0].max(dim=-1).values,
        "depth_input_min": depth.min(dim=-1).values,
        "depth_input_max": depth.max(dim=-1).values,
        "depth_input_mean": depth.mean(dim=-1),
    }
