from pathlib import Path

from omegaconf import OmegaConf


def get_project_path(path: str) -> Path:
    project_root = Path(__file__).parents[2] / path
    return project_root


def register_resolvers() -> None:
    OmegaConf.register_resolver("abs_path", get_project_path)
