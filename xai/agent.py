from . import *
from abc import ABC, abstractmethod
from dataclasses import dataclass

import dill # type: ignore
import copy

class Agent(ABC, Serializable["Agent"]):
    @dataclass
    class Step(ABC):
        observation: Observation
        action: Action
        rewards: Tuple[Reward,...]
        done:    bool

    def __init__(self) -> None:
        super().__init__()
        self.stats: Dict[str,Any] = {}

    @abstractmethod
    def rollout(self, *args: Any) -> Stream[Step]:
        pass

    def clone(self) -> Self:
        return copy.deepcopy(self)


        