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


def test_instinctagent_in_savanna_gym(test_hparams: DictConfig):
    params_savanna_gym = {
        "agent_id": "instinct_agent",
        "env": "savanna-gym-v2",
        "env_type": "gym",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 1,
        },
    }
    OmegaConf.merge(test_hparams, params_savanna_gym)
    run_episode(hparams=test_hparams, device="cpu")
    cleanup_gym_envs()
