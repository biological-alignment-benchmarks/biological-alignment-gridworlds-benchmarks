from typing import Tuple
import pathlib

from omegaconf import DictConfig, OmegaConf
import pytest


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture
def tparams_hparams(root_dir: pathlib.Path) -> Tuple[DictConfig, DictConfig]:
    full_params = OmegaConf.load(root_dir / "aintelope/config/config_experiment.yaml")
    tparams = full_params.trainer_params
    hparams = full_params.hparams
    return tparams, hparams
