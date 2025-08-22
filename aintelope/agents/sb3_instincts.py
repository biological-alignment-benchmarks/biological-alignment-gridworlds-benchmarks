# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import csv
import logging
import hashlib
from typing import List, Optional, Tuple
from collections import defaultdict
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from omegaconf import DictConfig

import numpy as np
import numpy.typing as npt

from aintelope.environments.savanna_safetygrid import (
    GridworldZooBaseEnv,
    ACTION_RELATIVE_COORDINATE_MAP,
)

from aintelope.agents.instincts.savanna_safetygrid_instincts import (
    savanna_safetygrid_available_instincts_dict,
    format_float,
)

from aintelope.agents.q_agent import QAgent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from typing import Union
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]

logger = logging.getLogger("aintelope.agents.sb3_instincts")


class SB3Instincts(object):
    """Instincts for SB3"""

    def __init__(
        self,
        env_classname: str,
        agent_id: str,
        cfg: DictConfig,
        target_instincts: List[str],
        action_space: spaces.Space,
    ) -> None:
        self.id = agent_id
        self.env_classname = env_classname
        self.cfg = cfg
        self.hparams = cfg.hparams
        # self.last_action = None
        self.action_space = action_space

        self.target_instincts = target_instincts
        self.instincts = {}

    def reset(self) -> None:
        self.init_instincts()

    # # NB! this method assumes that there is no hidden state that would affect the instinct action choice. Though it might be not a big deal either way.
    # # Currently this method is not actually used, because "deterministic" argument is set only during evaluation, not training.
    # def seed_from_observation(self, obs_np, round_decimals: int = 4):
    #     obs_rounded = np.round(obs_np, round_decimals)    # round so that states (potentially) differing only by floating point drift still get same tie breaking. This does not hurt (in case the states actually differ but are rounded to be equal), but makes the algorithm more robust in cases where the states differ only because of numeric stability issues.
    #     h = hashlib.blake2b(obs_rounded.tobytes(), digest_size=8, usedforsecurity=False).digest()  # blake2b is faster than sha1
    #     # h = hashlib.sha1(obs_rounded.tobytes(), usedforsecurity=False).digest()
    #     seed = int.from_bytes(h, "little", signed=False)
    #     return seed

    def tiebreaking_argmax(
        self,
        arr,
        _random=None,
    ):
        """Avoids the agent from repeatedly taking move-left action when the instinct tells the agent to move away from current cell in any direction. Then the instinct will not provide any q value difference in its q values for the different directions, they would be equal. Naive np.argmax would just return the index of first moving action, which happens to be always move-left action."""

        max_values_bitmap = np.isclose(arr, arr.max())
        max_values_indexes = np.flatnonzero(max_values_bitmap)

        if (
            len(max_values_indexes) == 0
        ):  # Happens when all values are infinities or nans. This would cause np.random.choice to throw.
            # result = np.random.randint(0, len(arr))
            result = int(_random.random() * len(arr))
        else:
            result = _random.choice(
                max_values_indexes
            )  # TODO: seed for this random generator

        return result

    def should_override(
        self,
        deterministic: bool = False,  # This is set only during evaluation, not training and the meaning is that the agent is greedy - it takes the best action. It does NOT mean that the action is always same.
        step: int = 0,
        env_layout_seed: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
        test_mode: bool = False,
        observation=None,
    ) -> int:
        # if deterministic:
        #     #  still use below random tiebreaking code, but use seed = ((pipeline_cycle * 10000 + episode) * 1000) + step
        #     # seed = ((pipeline_cycle * 10000 + episode) * 1000) + step
        #     seed = self.seed_from_observation(observation)
        #     _random = seeding.np_random(seed)[0]
        # else:
        #     _random = np.random
        _random = np.random

        if self.hparams.num_episodes == 0:  # there is no training
            # detected pure instinct agent test without training - if that config is detected then test mode will use the instincts
            return 1, _random  # instinct mode
        elif test_mode:
            return 0, None  # SB3 policy mode

        # TODO: warn if last_frame=0/1 or last_env_layout_seed=0/1 or last_episode=0/1 in any of the below values: for disabling the epsilon counting for corresponding variable one should use -1
        epsilon = (
            self.hparams.model_params.eps_start - self.hparams.model_params.eps_end
        )
        if self.hparams.model_params.eps_last_frame > 1:
            epsilon *= max(0, 1 - step / self.hparams.model_params.eps_last_frame)
        if self.hparams.model_params.eps_last_env_layout_seed > 1:
            epsilon *= max(
                0,
                1
                - env_layout_seed / self.hparams.model_params.eps_last_env_layout_seed,
            )
        if self.hparams.model_params.eps_last_episode > 1:
            epsilon *= max(0, 1 - episode / self.hparams.model_params.eps_last_episode)
        if self.hparams.model_params.eps_last_pipeline_cycle > 1:
            epsilon *= max(
                0,
                1 - pipeline_cycle / self.hparams.model_params.eps_last_pipeline_cycle,
            )
        epsilon += self.hparams.model_params.eps_end

        instinct_epsilon = (
            self.hparams.model_params.instinct_bias_epsilon_start
            - self.hparams.model_params.instinct_bias_epsilon_end
        )
        if self.hparams.model_params.eps_last_frame > 1:
            instinct_epsilon *= max(
                0, 1 - step / self.hparams.model_params.eps_last_frame
            )
        if self.hparams.model_params.eps_last_env_layout_seed > 1:
            instinct_epsilon *= max(
                0,
                1
                - env_layout_seed / self.hparams.model_params.eps_last_env_layout_seed,
            )
        if self.hparams.model_params.eps_last_episode > 1:
            instinct_epsilon *= max(
                0, 1 - episode / self.hparams.model_params.eps_last_episode
            )
        if self.hparams.model_params.eps_last_pipeline_cycle > 1:
            instinct_epsilon *= max(
                0,
                1 - pipeline_cycle / self.hparams.model_params.eps_last_pipeline_cycle,
            )
        instinct_epsilon += self.hparams.model_params.instinct_bias_epsilon_end

        apply_instinct_eps_before_random_eps = (
            self.hparams.model_params.apply_instinct_eps_before_random_eps
        )

        if (
            not apply_instinct_eps_before_random_eps
            and epsilon > 0
            and _random.random() < epsilon
        ):
            return 2, _random  # random exploration mode

        elif (
            instinct_epsilon > 0 and _random.random() < instinct_epsilon
        ):  # TODO: find a better way to combine epsilon and instinct_epsilon
            return 1, _random  # instinct mode

        elif (
            apply_instinct_eps_before_random_eps
            and epsilon > 0
            and _random.random() < epsilon
        ):
            return 2, _random  # random exploration mode

        else:
            return 0, None  # SB3 policy mode

    def get_action(
        self,
        observation=None,
        info: dict = {},
        step: int = 0,
        env_layout_seed: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
        test_mode: bool = False,
        override_type: int = 0,
        deterministic: bool = False,  # This is set only during evaluation, not training and the meaning is that the agent is greedy - it takes the best action. It does NOT mean that the action is always same.
        _random=None,
    ) -> Optional[int]:
        """Given an observation, ask your rules what to do.

        Returns:
            action (Optional[int]): index of action
        """

        action_space = self.action_space
        if isinstance(action_space, Discrete):
            min_action = action_space.start
            max_action = action_space.start + action_space.n - 1
        else:
            min_action = action_space.min_action
            max_action = action_space.max_action

        # calculate action reward predictions using instincts
        action_rewards = defaultdict(float)

        for instinct_name, instinct_object in self.instincts.items():
            instinct_action_rewards = {}
            # predict reward for all available actions
            for action in range(
                min_action, max_action + 1
            ):  # NB! max_action is inclusive max
                agent_coordinate = info[ACTION_RELATIVE_COORDINATE_MAP][action]

                (
                    instinct_reward,
                    instinct_event,
                ) = instinct_object.calc_reward(
                    self,
                    observation,
                    info,
                    agent_coordinate=agent_coordinate,
                    predicting=True,
                )

                instinct_action_rewards[action] = instinct_reward
                action_rewards[action] += instinct_reward  # TODO: nonlinear aggregation

            # debug helper  # TODO: refactor into a separate method
            # if instinct_name == "gold":
            #    q_values = np.zeros([max_action - min_action + 1], np.float32)
            #    for action, bias in instinct_action_rewards.items():
            #        q_values[action - min_action] = bias
            #    print(f"gold q_values: {format_float(q_values)}")

        instinct_q_values = action_rewards  # instincts see only one step ahead

        if override_type == 2:
            action = action_space.sample()

        elif (
            override_type == 1
        ):  # TODO: find a better way to combine epsilon and instinct_epsilon
            q_values = np.zeros([max_action - min_action + 1], np.float32)
            for action, q_value in instinct_q_values.items():
                q_values[action - min_action] = q_value
            action = (
                self.tiebreaking_argmax(
                    q_values,
                    _random,
                )
                + min_action
            )  # take best action predicted by instincts

        else:
            action = None

        # print(f"Action: {action}")
        # self.last_action = action
        return action

    def init_instincts(self) -> None:
        if self.env_classname in [
            "aintelope.environments.savanna_safetygrid.GridworldZooBaseEnv"
        ]:  # radically different types of environments may need different instincts
            available_instincts_dict_local = savanna_safetygrid_available_instincts_dict

        logger.debug(f"target_instincts: {self.target_instincts}")
        for instinct_name in self.target_instincts:
            if instinct_name not in available_instincts_dict_local:
                logger.warning(
                    f"Warning: could not find {instinct_name} "
                    "in available_instincts_dict"
                )
                continue

        self.instincts = {
            instinct: available_instincts_dict_local.get(instinct)()
            for instinct in self.target_instincts
            if instinct in available_instincts_dict_local
        }
        for instinct in self.instincts.values():
            instinct.reset()
