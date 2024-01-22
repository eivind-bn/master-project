from typing import *
from typing import Any
from torch import Tensor
from .feed_forward import FeedForward

import torch

Device = Literal["cpu", "cuda", "cuda:0", "cuda:1", "auto"]
Optimizer = Literal["sgd", "adam"]

class Policy:

    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: int|Sequence[int] = 2,
                 device: Device = "auto") -> None:
        super().__init__()

        if device == "auto":
            self.device: Device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._net = torch.nn.Sequential()

        I, O = input_dim, output_dim

        if isinstance(hidden_layers, int):
            N = hidden_layers + 1
            layers = [I] + [int(I - ((I - O)*n)/(N)) for n in range(1, hidden_layers+1)] + [output_dim]
        elif isinstance(hidden_layers, Sequence):
            layers = [I] + list(hidden_layers) + [O]
        else:
            raise TypeError(f"Argument for layers has incompatible type: {type(hidden_layers)}")
        
        for i in range(1, len(layers)-1):
            self._net += torch.nn.Sequential(
                torch.nn.Linear(layers[i-1], layers[i], device=self.device),
                torch.nn.ReLU()
            )

        self._net += torch.nn.Sequential(torch.nn.Linear(layers[-2], layers[-1], device=self.device))
        
    def predict(self, x: Tensor|FeedForward) -> FeedForward:
        if isinstance(x, Tensor):
            x = x.to(self.device)
        else:
            x = x._output.to(self.device)

        return FeedForward(
            input=x,
            network=self._net
        )
    def __call__(self, x: Tensor|FeedForward) -> FeedForward:
        return self.predict(x)
    
    def fit(self, 
            x:              Tensor, 
            y:              Tensor, 
            steps:          int, 
            learning_rate:  float, 
            optimizer:      Optimizer) -> None:
        match optimizer:
            case "sgd":
                self._sgd_fit()
            case "adam":
                self._adam_fit()
            case _:
                raise ValueError(f"Optimizer: {optimizer} is invalid.")
            
    def save(self, path: str) -> None:
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "Policy":
        policy: Policy = torch.load(path)
        if isinstance(policy, Policy):
            return policy
        else:
            raise TypeError(f"Unpickled object is not a Policy, but of type: {type(policy)}")
