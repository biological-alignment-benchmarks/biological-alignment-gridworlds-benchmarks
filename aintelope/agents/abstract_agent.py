from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy.typing as npt
from aintelope.typing import ObservationFloat
from pettingzoo import AECEnv, ParallelEnv

Environment = Union[AECEnv, ParallelEnv]


class Agent(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def get_action(self, epsilon: float, device: str) -> Optional[int]:
        ...

    @abstractmethod
    def update(
        self,
        env: Environment,
        observation: npt.NDArray[ObservationFloat],
        score: float,
        done: bool,
        save_path: Optional[str],
    ) -> list:
        ...
