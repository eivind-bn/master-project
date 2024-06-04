from . import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.nn import Module

import copy

Ints: TypeAlias = Tuple[int,...]
Sx = TypeVar("Sx", bound=Ints)
Sy = TypeVar("Sy", bound=Ints)

class Agent(Network[Sx,Sy]):
    @dataclass
    class Step(ABC):
        observation: Observation
        action: Action
        rewards: Tuple[Reward,...]
        done:    bool

    def __init__(self, 
                 device:        Device, 
                 input_shape:   Sx, 
                 output_shape:  Sy,
                 logits:        Module|None = None) -> None:
        super().__init__(
            device=device,
            input_shape=input_shape,
            output_shape=output_shape,
            logits=logits
        )
        self.stats: Dict[str,Any] = {}

    @abstractmethod
    def rollout(self, *args: Any) -> Stream[Step]:
        pass

    def clone(self) -> Self:
        return copy.deepcopy(self)


        