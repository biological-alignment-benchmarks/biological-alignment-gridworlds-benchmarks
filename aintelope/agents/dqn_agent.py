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
from aintelope.environments.singleagent_zoo_to_gym_wrapper import (
    SingleAgentZooToGymWrapper,
)

import torch
from stable_baselines3 import DQN
import supersuit as ss

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.dqn_agent")


# need separate function outside of class in order to init multi-model training threads
def dqn_model_constructor(env, cfg):
    # policy_kwarg:
    # if you want to use CnnPolicy or MultiInputPolicy with image-like observation (3D tensor) that are already normalized, you must pass normalize_images=False
    # see the following links:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    # https://github.com/DLR-RM/stable-baselines3/issues/1863
    # Also: make sure your image is in the channel-first format.
    return DQN(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs={
            "normalize_images": False,
            "features_extractor_class": CustomCNN,  # need custom CNN in order to handle observation shape 9x9
            "features_extractor_kwargs": {
                "features_dim": 256,  # TODO: config parameter. Note this is not related to the number of features in the original observation (15), this parameter here is model's internal feature dimensionality
                "num_conv_layers": cfg.hparams.model_params.num_conv_layers,
            },
        },
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),  # Note, CUDA-based CPU performance is much better than Torch-CPU mode.
    )


class DQNAgent(SB3BaseAgent):
    """DQNAgent class from stable baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

    """

    def __init__(
        self,
        env: PettingZooEnv = None,
        cfg: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__(env=env, cfg=cfg, **kwargs)

        self.model_constructor = dqn_model_constructor

        if (
            self.env.num_agents == 1 or self.test_mode
        ):  # during test, each agent has a separate in-process instance with its own model and not using threads/subprocesses
            env = SingleAgentZooToGymWrapper(env, self.id)
            self.model = self.model_constructor(env, cfg)
        else:
            pass  # multi-model training will be automatically set up by the base class when self.model is None. These models will be saved to self.models and there will be only one agent instance in the main process. Actual agents will run in threads/subprocesses because SB3 requires Gym interface.

    # this method is currently called only in test mode
    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        super().reset(state, info, env_class)

    def get_action(self, *args, **kwargs) -> Optional[int]:
        """Given an observation, ask your net what to do. State is needed to be
        given here as other agents have changed the state!

        Returns:
            action (Optional[int]): index of action
        """
        action = super().get_action(*args, **kwargs)
        return action

    def update(self, *args, **kwargs) -> list:
        """
        Takes observations and updates trainer on perceived experiences.
        Needed here to catch instincts.

        Args:
            env: Environment
            observation: Tuple[ObservationArray, ObservationArray]
            score: Only baseline uses score as a reward
            done: boolean whether run is done
            save_path: str
        Returns:
            agent_id (str): same as elsewhere ("agent_0" among them)
            state (Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]]): input for the net
            action (int): index of action
            reward (float): reward signal
            done (bool): if agent is done
            next_state (npt.NDArray[ObservationFloat]): input for the net
        """
        event = super().update(*args, **kwargs)
        return event

    def train(self, steps):
        super().train(steps)

    def save_model(self):
        super().save_model()

    def load_model(self, checkpoint):
        super().load_model(checkpoint)
