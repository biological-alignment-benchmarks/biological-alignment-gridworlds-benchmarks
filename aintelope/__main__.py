import hydra
from omegaconf import DictConfig, OmegaConf

from aintelope.training.dqn_lightning_trainer import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def aintelope_main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_experiment(**cfg)


if __name__ == "__main__":
    aintelope_main()
