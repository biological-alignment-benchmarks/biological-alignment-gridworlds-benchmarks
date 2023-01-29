import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers
from aintelope.training.lightning_trainer import run_experiment

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_main")
def aintelope_main(cfg: DictConfig) -> None:
    logger.info("Running training with the following configuration")
    logger.info(OmegaConf.to_yaml(cfg))
    run_experiment(cfg.training)


if __name__ == "__main__":
    register_resolvers()
    aintelope_main()
