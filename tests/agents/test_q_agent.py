import pathlib

from omegaconf import OmegaConf, DictConfig
import pytest

from aintelope.environments.env_utils.cleanup import cleanup_gym_envs
from aintelope.training.simple_eval import run_episode
from tests.test_config import root_dir


@pytest.fixture
def test_hparams(root_dir: pathlib.Path) -> DictConfig:
    full_params = OmegaConf.load(root_dir / "aintelope/config/config_experiment.yaml")
    hparams = full_params.hparams
    return hparams


def test_qagent_in_savanna_zoo_sequential(test_hparams: DictConfig):
    params_zoo_sequential = {
        "agent_id": "q_agent",
        "env": "savanna-zoo-sequential-v2",
        "env_entry_point": None,
        "env_type": "zoo",
        "sequential_env": True,
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
        },
        "agent_params": {},
    }
    OmegaConf.merge(test_hparams, params_zoo_sequential)
    run_episode(hparams=test_hparams, device="cpu")


def test_qagent_in_savanna_zoo_parallel(test_hparams: DictConfig):
    params_zoo_parallel = {
        "agent_id": "q_agent",
        "env": "savanna-zoo-parallel-v2",
        "env_entry_point": None,
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
        },
        "agent_params": {},
    }
    OmegaConf.merge(test_hparams, params_zoo_parallel)
    run_episode(hparams=test_hparams, device="cpu")


def test_qagent_in_savanna_gym(test_hparams: DictConfig):
    params_savanna_gym = {
        "agent_id": "q_agent",
        "env": "savanna-gym-v2",
        "env_type": "gym",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
        },
        "agent_params": {},
    }
    OmegaConf.merge(test_hparams, params_savanna_gym)
    run_episode(hparams=test_hparams, device="cpu")
    cleanup_gym_envs()
