from typing import *
from abc import ABC, abstractmethod

from torch.nn.functional import mse_loss, huber_loss, cross_entropy, binary_cross_entropy

class Loss:
    
    def __init__(self) -> None:
        pass

class MseLoss(Loss):
    pass

class HuberLoss(Loss):
    pass

class CrossEntropyLoss(Loss):
    pass

class BinaryCrossEntropyLoss(Loss):
    pass