import logging

import numpy as np
import gym
from gym.spaces import Box, Discrete
from gym.utils import seeding

from aintelope.environments.env_utils.render_ascii import AsciiRenderState
from aintelope.environments.env_utils.distance import distance_to_closest_item
from aintelope.environments.savanna import (
    SavannaEnv,
    RenderSettings,
    RenderState,
    move_agent,
    reward_agent,
    PositionFloat,
    Action,
)

logger = logging.getLogger("aintelope.environments.savanna_gym")


class SavannaGymEnv(SavannaEnv, gym.Env):
    metadata = {
        "name": "savanna-v2",
        "render_fps": 3,
        "render_agent_radius": 5,
        "render_agent_color": (200, 50, 0),
        "render_grass_radius": 5,
        "render_grass_color": (20, 200, 0),
        "render_modes": ("human", "ascii", "offline"),
        "render_window_size": 512,
    }

    def __init__(self, env_params={}):
        SavannaEnv.__init__(self, env_params)
        gym.Env.__init__(self)
        assert self.metadata["amount_agents"] == 1, "agents must == 1 for gym env"

    def step(self, action):
        actions = {self._agent_id: action}
        # should be: observations, rewards, dones, infos
        # but per agent
        res = SavannaEnv.step(self, actions)

        observations, rewards, dones, infos = res
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(infos, dict)

        # so just return the first
        res = tuple(r[self._agent_id] if isinstance(r, dict) else r for r in res)
        logger.warning(res)
        # should return observation, reward, done, info
        return res

    def reset(self, seed=None, options={}):
        observations = SavannaEnv.reset(self, seed, options)
        # FIXME: infos are additional information for the agent, like some position etc.
        info = {"placeholder": "hmmm"}
        return (observations[self._agent_id], info)

    @property
    def _agent_id(self):
        return self.possible_agents[0]

    @property
    def action_space(self):
        return self._action_spaces[self._agent_id]

    @property
    def observation_space(self):
        return self._observation_spaces[self._agent_id]
