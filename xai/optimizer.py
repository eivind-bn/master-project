from typing import *
from dataclasses import dataclass
from typing import Any
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer, SGD, Adam
from .policy import Policy

import torch

T = TypeVar("T", bound=Optimizer)

class OptimizerFactory(Generic[T]):
    
    def __init__(self, cls: Type[T], *args: Any) -> None:
        self.optimizer_cls = cls
        self.args = args

    def __call__(self, params: Iterable[Parameter]) -> T:
        return self.optimizer_cls(params, *self.args)

class Optimizers:

    @staticmethod
    def sgd(lr:             float,
            momentum:       float|None = None,
            dampening:      float|None = None,
            weight_decay:   float|None = None,
            nesterov:       bool|None = None) -> OptimizerFactory[SGD]:
        return OptimizerFactory(cls=SGD, **locals())
    
    @staticmethod
    def adam(lr:             float|Tensor|None = None,
             weight_decay:   float|None = None,
             betas:          Tuple[float,float]|None = None,
             eps:            float|None = None,
             amsgrad:        bool|None = None,
             foreach:        bool|None = None,
             maximize:       bool|None = None,
             capturable:     bool|None = None,
             differentiable: bool|None = None,
             fused:          bool|None = None) -> OptimizerFactory[Adam]:
        return OptimizerFactory(cls=Adam, **locals())