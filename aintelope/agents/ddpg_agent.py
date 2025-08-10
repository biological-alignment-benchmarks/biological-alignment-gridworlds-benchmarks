# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import logging
from typing import List, NamedTuple, Optional, Tuple
from gymnasium.spaces import Discrete

import pandas as pd
from omegaconf import DictConfig

from aintelope.utils import RobustProgressBar

import numpy as np
import numpy.typing as npt
import os
import datetime

from aintelope.agents.sb3_base_agent import (
    SB3BaseAgent,
    CustomCNN,
)
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer
from zoo_to_gym_multiagent_adapter.singleagent_zoo_to_gym_adapter import (
    SingleAgentZooToGymAdapter,
)

import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import supersuit as ss

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.ddpg_agent")


# need separate function outside of class in order to init multi-model training threads
def ddpg_model_constructor(env, cfg):
    n_actions = (
        env.action_space.n
    )  # Use self.env to get access to original Zoo env API. In contrast, the env variable contains a Gym env with a different API.
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),  # TODO: config parameter for sigma
    )

    # policy_kwarg:
    # if you want to use CnnPolicy or MultiInputPolicy with image-like observation (3D tensor) that are already normalized, you must pass normalize_images=False
    # see the following links:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    # https://github.com/DLR-RM/stable-baselines3/issues/1863
    # Also: make sure your image is in the channel-first format.
    return DDPG(
        "CnnPolicy" if cfg.hparams.model_params.num_conv_layers > 0 else "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        policy_kwargs=(
            {
                "normalize_images": False,
                "features_extractor_class": CustomCNN,  # need custom CNN in order to handle observation shape 9x9
                "features_extractor_kwargs": {
                    "features_dim": 256,  # TODO: config parameter. Note this is not related to the number of features in the original observation (15 or 39), this parameter here is model's internal feature dimensionality
                    "num_conv_layers": cfg.hparams.model_params.num_conv_layers,
                },
            }
            if cfg.hparams.model_params.num_conv_layers > 0
            else {"normalize_images": False}
        ),
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),  # Note, CUDA-based CPU performance is much better than Torch-CPU mode.
        tensorboard_log=cfg.tensorboard_dir,
        optimize_memory_usage=True,
        replay_buffer_kwargs={
            "handle_timeout_termination": False
        },  # handle_timeout_termination has to be False if optimize_memory_usage = True. Because test episodes have same length as training episodes, we can correctly disable timeout termination handling here.
        # TODO: add a remaining-time feature to the input, using TimeFeatureWrapper from sb3_contrib
    )


class DDPGAgent(SB3BaseAgent):
    """DDPGAgent class from stable baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html

    """

    def __init__(
        self,
        env: PettingZooEnv = None,
        cfg: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__(env=env, cfg=cfg, **kwargs)

        self.model_constructor = ddpg_model_constructor

        if (
            self.env.num_agents == 1 or self.test_mode
        ):  # during test, each agent has a separate in-process instance with its own model and not using threads/subprocesses
            env = SingleAgentZooToGymAdapter(env, self.id)
            # TODO: Action space wrapper since this model expects Box action space, not Discrete
            self.model = self.model_constructor(env, cfg)
        else:
            pass  # multi-model training will be automatically set up by the base class when self.model is None. These models will be saved to self.models and there will be only one agent instance in the main process. Actual agents will run in threads/subprocesses because SB3 requires Gym interface.
