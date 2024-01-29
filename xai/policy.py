from typing import *
from typing import Any
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Sequential, Module, Parameter
from torch.optim import Optimizer, SGD, Adam
from .feed_forward import FeedForward
from .optimizer import OptimizerFactory, Optimizers

import torch
import copy

Device = Literal["cpu", "cuda", "cuda:0", "cuda:1", "auto"]
OptimizerType = Literal["sgd", "adam"]
LossType = Literal["mse", "huber"]

@dataclass(frozen=True)
class Policy:
    input_dim:      int
    output_dim:     int
    device:         Device
    network:        Sequential
    optimizer:      OptimizerFactory

    def predict(self, 
                x:              Tensor|FeedForward, 
                move_to_device: bool = False) -> FeedForward:
        
        if isinstance(x, FeedForward):
            x = x._output

        if move_to_device:
            x = x.to(device=self.device)

        return FeedForward(
            input=x,
            network=self.network
        )
    
    def parameters(self) -> Iterator[Parameter]:
        return self.network.parameters()
            
    def save(self, path: str) -> None:
        torch.save(self, path)

    def copy(self) -> "Policy":
        return copy.deepcopy(self)

    @staticmethod
    def load(path: str) -> "Policy"|NoReturn:
        policy: Policy = torch.load(path)
        if isinstance(policy, Policy):
            return policy
        else:
            raise TypeError(f"Unpickled object is not a Policy, but of type: {type(policy)}")
        
    @staticmethod
    def new(input_dim:      int,
            output_dim:     int,
            optimizer:      Callable[[Type[Optimizers]],OptimizerFactory],
            hidden_layers:  int|Sequence[int] = 2,
            device:         Device = "auto") -> "Policy":
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        network = torch.nn.Sequential()

        I, O = input_dim, output_dim

        if isinstance(hidden_layers, int):
            N = hidden_layers + 1
            layers = [I] + [int(I - ((I - O)*n)/(N)) for n in range(1, hidden_layers+1)] + [output_dim]
        elif isinstance(hidden_layers, Sequence):
            layers = [I] + list(hidden_layers) + [O]
        else:
            raise TypeError(f"Argument for layers has incompatible type: {type(hidden_layers)}")
        
        for i in range(1, len(layers)-1):
            network += torch.nn.Sequential(
                torch.nn.Linear(layers[i-1], layers[i], device=device),
                torch.nn.ReLU()
            )

        network += torch.nn.Sequential(torch.nn.Linear(layers[-2], layers[-1], device=device))

        return Policy(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            network=network,
            optimizer=optimizer(Optimizers)
        )
    
    def __add__(self, other: "Policy") -> "Policy":
        assert self.output_dim == other.input_dim
        return Policy(
            input_dim=self.input_dim,
            output_dim=other.output_dim,
            device=self.device,
            network=self.network + other.network,
            optimizer=None
        )
    
    def __call__(self, x: Tensor|FeedForward) -> FeedForward:
        return self.predict(x)
    
    def __iter__(self) -> Iterator[Parameter]:
        return self.parameters()
    
    def __repr__(self) -> str:
        return str(self.network)
