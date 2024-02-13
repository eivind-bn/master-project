from typing import *
from abc import ABC, abstractmethod
from collections import deque
from .observation import Observation
from .action import Action
from .fitness import NormalizedFitness

import dill
import copy

class Agent(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.stats: Dict[str,Any] = {}

    @abstractmethod
    def predict(self, observation: Observation) -> Action:
        pass

    @abstractmethod
    def breed(self,               
              partners:         Iterable[Self], 
              volatility:       float,
              mutation_rate:    float) -> Self:
        pass

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            dill.dump(self, file)

    def clone(self) -> Self:
        return copy.deepcopy(self)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as file:
            dqn: Self = dill.load(file)

        if isinstance(dqn, cls):
            return dqn
        else:
            raise TypeError(f"Invalid type: {type(dqn)}")
        