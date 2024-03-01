from typing import *
from abc import ABC, abstractmethod
from .observation import Observation
from .action import Action
from .window import Window
from .record import Recorder
from .asteroids import Asteroids

import pickle
import copy

class Agent(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.stats: Dict[str,Any] = {}

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def clone(self) -> Self:
        return copy.deepcopy(self)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as file:
            dqn: Self = pickle.load(file)

        if isinstance(dqn, cls):
            return dqn
        else:
            raise TypeError(f"Invalid type: {type(dqn)}")


        