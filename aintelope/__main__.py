import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import register_resolvers
from aintelope.experiments import run_experiment

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def aintelope_main(cfg: DictConfig) -> None:
    pipeline_config = OmegaConf.load("aintelope/config/config_pipeline.yaml")
    for env_conf in pipeline_config:
        OmegaConf.update(cfg, "experiment_name", env_conf)
        OmegaConf.update(cfg, "hparams", pipeline_config[env_conf], force_add=True)
        logger.info("Running training with the following configuration")
        logger.info(OmegaConf.to_yaml(cfg))
        run_experiment(cfg)
    analytics(cfg)


def analytics(cfg):
    savepath = cfg.log_dir + "plot.png"
    events = recording.read_events(cfg.log_dir, cfg.events_dir)
    plotting.plot_performance(events, savepath)


if __name__ == "__main__":
    register_resolvers()
    aintelope_main()
