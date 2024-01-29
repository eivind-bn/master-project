from typing import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import Tensor, Module
from torch.nn.functional import (mse_loss, 
                                 huber_loss, 
                                 cross_entropy, 
                                 binary_cross_entropy)

@dataclass(frozen=True)
class Loss(ABC):
    loss_criterion: Module
    
    def fit(self,) -> None:
        binary_cross_entropy()

class MseLoss(Loss):
    
    def __init__(self, X: Tensor, Y: Tensor) -> None:
        super().__init__()

class HuberLoss(Loss):
    pass

class CrossEntropyLoss(Loss):
    pass

class BinaryCrossEntropyLoss(Loss):
    pass