import pathlib

import pytest


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]
