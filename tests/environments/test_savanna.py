import pytest
import numpy as np
import numpy.testing as npt
from gym.spaces import Discrete


from aintelope.environments import savanna as sut

# Base Savanna is not yet PettingZoo or Gym env
# just the shared elements those will depend on


def test_max_cycles():
    # currently the environment does not accept parameters like max_cycles
    # max_cycles_test(sut.env)
    pass


def test_render():
    # TODO: close method not implemented
    # render_test(sut.env)
    pass


def test_step_result():
    env = sut.env()
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.reset()

    agent = env.possible_agents[0]
    action = {agent: env.action_space(agent).sample()}
    observations, rewards, dones, info = env.step(action)

    assert not dones[agent]
    assert isinstance(observations, dict), "observations is not a dict"
    assert isinstance(
        observations[agent], np.ndarray
    ), "observations of agent is not an array"
    assert isinstance(rewards, dict), "rewards is not a dict"
    assert isinstance(
        rewards[agent], np.float64
    ), "reward of agent is not a float64"


def test_grass_patches():
    env = sut.SavannaEnv()

    with pytest.raises(AttributeError):
        env.grass_patches
    with pytest.raises(AttributeError):
        env.grass_patches

    env.reset()
    assert len(env.grass_patches) == env.metadata['amount_grass_patches']
    assert isinstance(env.grass_patches, np.ndarray)
    assert env.grass_patches.shape[1] == 2


def test_observation_spaces():
    pass  # TODO
