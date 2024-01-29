from typing import *
from torch import Tensor
from dataclasses import dataclass
from torch.nn import Sequential, Module

import torch

class Choices:
    
    def __init__(self, values: Tensor) -> None:
        assert not values.is_floating_point() and values.dim() == 1, f"Invalid datatype: {values.dtype}"
        self._values = values

    def tensor(self) -> Tensor:
        return self._values
    
    def tuple(self) -> Tuple[int,...]:
        return tuple(self)
    
    def list(self) -> List[int]:
        return self._values.tolist()
    
    def __iter__(self) -> Iterator[int]:
        return iter(self._values.tolist())
    
    def __repr__(self) -> str:
        return str(self._values)
    
LossType = Literal["mse", "huber", "cross_entropy"]

class FeedForward:

    def __init__(self, 
                 input:     Tensor,
                 network:   Sequential,
                 loss_function: Callable[[Tensor,Tensor],Tensor]) -> None:
           
        self._loss_function = loss_function
        self._network = network
        self._input = input.detach().requires_grad_(True)
        self._output: Tensor = self._network(self._input)
        self._input = input.requires_grad_(False)
        self._gradients: Tensor|None = None

    def greedy_max(self) -> Choices:
        return Choices(self._output.argmax(dim=1))
    
    def greedy_min(self) -> Choices:
        return Choices(self._output.argmin(dim=1))
    
    def random(self) -> Choices:
        return Choices(torch.randint(
            low=0, 
            high=self._output.shape[1], 
            size=(self._output.shape[0],),
            device=self._output.device
            ))
    
    def epsilon_greedy(self, epsilon: float) -> Choices:
        assert 0.0 < epsilon < 1.0, f"Epsilon is not: 0 < {epsilon} < 1.0"
        with torch.no_grad():
            greedy_choices = self.greedy_max().tensor()
            random_choices = self.random().tensor()
            rand = torch.rand(greedy_choices.size(), device=greedy_choices.device)
            epsilon_greedy = torch.where(rand < epsilon, random_choices, greedy_choices)
        return Choices(epsilon_greedy)
    
    def gradients(self, target: Sequence[int]) -> Tensor:
        assert len(target) == self._output.shape[0]
        if self._gradients is None:
            requires_grad = self._input.requires_grad
            self._input.requires_grad = True
            idx = torch.arange(0, self._output.shape[0])
            choices = self._output[idx,target]
            self._gradients = torch.autograd.grad(
                outputs=choices,
                inputs=self._input,
                grad_outputs=torch.ones_like(choices)
            )[0]
            self._input.requires_grad = requires_grad
        
        return self._gradients
    
    def loss(self, target: Tensor) -> Tensor:
        return self._loss_function(self._output, target)
    
    def tensor(self) -> Tensor:
        return self._output
    
    def __repr__(self) -> str:
        return str(self._output)