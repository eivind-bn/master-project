from . import *
from abc import ABC

import dill
import copy

class Agent(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.stats: Dict[str,Any] = {}

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


        