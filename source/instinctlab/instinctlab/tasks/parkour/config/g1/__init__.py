# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.g1"

gym.register(
    id="Instinct-Parkour-Curb-Crossing-Amp-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.curb_crossing_cfg:G1CurbCrossingEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.curb_crossing_rl_cfg:G1CurbCrossingPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-Mixed-Amp-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.mixed_parkour_cfg:G1MixedParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.mixed_parkour_rl_cfg:G1MixedParkourPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-Mixed-Amp-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.mixed_parkour_cfg:G1MixedParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.mixed_parkour_rl_cfg:G1MixedParkourPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-Stairs-Amp-G1-v1",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.stair_training_cfg:G1StairTrainingEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.stair_training_rl_cfg:G1StairPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)
