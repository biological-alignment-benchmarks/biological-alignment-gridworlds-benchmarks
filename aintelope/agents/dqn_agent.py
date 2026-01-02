# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

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
    get_optimizer_class,
    PolicyWithConfigFactory,
    INFO_PIPELINE_CYCLE,
    INFO_EPISODE,
    INFO_ENV_LAYOUT_SEED,
    INFO_STEP,
    INFO_TEST_MODE,
)
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer
from aintelope.agents.sb3_handwritten_rules_expert import SB3HandWrittenRulesExpert
from zoo_to_gym_multiagent_adapter.singleagent_zoo_to_gym_adapter import (
    SingleAgentZooToGymAdapter,
)

import torch
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.type_aliases import PyTorchObs
import supersuit as ss

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.dqn_agent")


class ExpertOverrideMixin:
    def __init__(self, env_classname, agent_id, cfg, *args, **kwargs):
        self.cfg = cfg
        super().__init__(*args, **kwargs)

        self.expert = SB3HandWrittenRulesExpert(
            env_classname=env_classname,
            agent_id=agent_id,
            cfg=cfg,
            action_space=self.action_space,
            **cfg.hparams.agent_params,
        )

    def set_info(self, info):
        self.info = info

    def my_reset(self, observation, info):
        self.info = info
        self.expert.reset()

    # code adapted from
    # https://github.com/DLR-RM/stable-baselines3/blob/dd7f5bfe63631630463f2f9bcb4762e6c040de12/stable_baselines3/dqn/policies.py#L183
    @torch.no_grad()
    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        actions = self.q_net._predict(obs, deterministic=deterministic)

        # inserted code begins

        step = self.info[INFO_STEP]
        env_layout_seed = self.info[INFO_ENV_LAYOUT_SEED]
        episode = self.info[INFO_EPISODE]
        pipeline_cycle = self.info[INFO_PIPELINE_CYCLE]
        test_mode = self.info[INFO_TEST_MODE]

        obs_nps = obs.detach().cpu().numpy()
        obs_np = obs_nps[0, :]

        (override_type, _random) = self.expert.should_override(
            deterministic,
            step,
            env_layout_seed,
            episode,
            pipeline_cycle,
            test_mode,
            0,  # agent_i
            obs_np,
        )
        if override_type != 0:
            action = self.expert.get_action(
                obs_np,
                self.info,
                step,
                env_layout_seed,
                episode,
                pipeline_cycle,
                test_mode,
                0,  # agent_i
                override_type,
                deterministic,
                _random,
            )
            # TODO: handle multiple observations and actions (for that we need also multiple infos)
            # TODO: option to softly change the actions Q values, not actions directly, so that actions less prioritised according to model's Q values get still chosen less often
            actions = [action]
            # actions = torch.as_tensor(actions, device=obs.device, dtype=torch.long)
            # optimization: Use None device, as the action will be transferred to CPU in the call site anyway
            # see https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L370
            # device: If None and data is not a tensor then the result tensor is constructed on the current device.
            actions = torch.as_tensor(actions, device=None, dtype=torch.long)

        # inserted code ends

        return actions


class CnnPolicyWithExpertOverride(ExpertOverrideMixin, CnnPolicy):
    pass


class MlpPolicyWithExpertOverride(ExpertOverrideMixin, MlpPolicy):
    pass


# need separate function outside of class in order to init multi-model training threads
def dqn_model_constructor(env, env_classname, agent_id, cfg):
    # policy_kwarg:
    # if you want to use CnnPolicy or MultiInputPolicy with image-like observation (3D tensor) that are already normalized, you must pass normalize_images=False
    # see the following links:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    # https://github.com/DLR-RM/stable-baselines3/issues/1863
    # Also: make sure your image is in the channel-first format

    use_imitation_learning = (
        cfg.hparams.model_params.instinct_bias_epsilon_start > 0
        or cfg.hparams.model_params.instinct_bias_epsilon_end > 0
    )
    if use_imitation_learning:
        policy_override_class = (
            CnnPolicyWithExpertOverride
            if cfg.hparams.model_params.num_conv_layers > 0
            else MlpPolicyWithExpertOverride
        )
        policy = PolicyWithConfigFactory(
            env_classname, agent_id, cfg, policy_override_class
        )
    else:
        policy = (
            "CnnPolicy" if cfg.hparams.model_params.num_conv_layers > 0 else "MlpPolicy"
        )

    policy_kwargs = (
        {
            "normalize_images": False,
            "features_extractor_class": CustomCNN,  # need custom CNN in order to handle observation shape 9x9
            "features_extractor_kwargs": {
                "features_dim": 256,  # TODO: config parameter. Note this is not related to the number of features in the original observation (15 or 39), this parameter here is model's internal feature dimensionality
                "num_conv_layers": cfg.hparams.model_params.num_conv_layers,
            },
            # DQN does not have "use_expln" argument
        }
        if cfg.hparams.model_params.num_conv_layers > 0
        else {
            "normalize_images": False,
            # DQN does not have "use_expln" argument
        }
    )

    # optimiser: for compatibility with PPO and A2C, probably not needed here.
    # For PPO it was added to enable AdamW optimizer to avoid NaNs in SB3 tensors - see https://github.com/DLR-RM/rl-baselines3-zoo/issues/427#issuecomment-1829495239
    # DQN does not have "use_expln" argument
    optimizer_class = get_optimizer_class(cfg.hparams.model_params.optimizer_class)
    if (
        optimizer_class is not None
    ):  # cannot specify None as dictionary entry, else SB3 would fail
        policy_kwargs["optimizer_class"] = optimizer_class

    return DQN(
        policy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
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
            # TODO: Environment duplication support for parallel compute purposes. Abseil package needs to be replaced for that end. Also note that environment seeding needs to be adapted so that each environment gets potentially a different seed, as it is currently set by experiments.py.
            adapter_env = SingleAgentZooToGymAdapter(env, self.id)
            if cfg.hparams.model_params.early_detect_nans:
                # VecCheckNan expects a vectorised env. The reset() method of vectorised env does not return a tuple like Gym env does. Also, the step infos are provided inside a list (though infos are not used or checked).
                env = DummyVecEnv([lambda: adapter_env])
                env = VecCheckNan(env, raise_exception=True)
            else:
                env = adapter_env
            self.model = self.model_constructor(env, self.env_classname, self.id, cfg)
        else:
            pass  # multi-model training will be automatically set up by the base class when self.model is None. These models will be saved to self.models and there will be only one agent instance in the main process. Actual agents will run in threads/subprocesses because SB3 requires Gym interface.
