from typing import *
from torch import Tensor
from torch.nn import Sequential

import torch

class FeedForward:

    def __init__(self, 
                 input:     Tensor,
                 network:   Sequential) -> None:
        
        self._input = input.requires_grad_(True)
        self._output: Tensor = network(self._input)
        self._input = input.requires_grad_(False)
        self._gradients: Tensor|None = None

    def greedy_max(self) -> Tensor:
        return self._output.unsqueeze(0).argmax(dim=1)
    
    def greedy_min(self) -> Tensor:
        return torch.argmax(self._output, dim=1)
    
    def gradients(self, target: int) -> Tensor:
        if self._gradients is None:
            requires_grad = self._input.requires_grad
            self._input.requires_grad = True
            self._gradients = torch.autograd.grad(
                outputs=self._output[target],
                inputs=self._input
            )[0]
            self._input.requires_grad = requires_grad
        
        return self._gradients
    
    def __repr__(self) -> str:
        return str(self._output)