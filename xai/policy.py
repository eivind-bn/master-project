from typing import *
from torch import Tensor

import torch

Device = Literal["cpu", "cuda", "cuda:0", "cuda:1", "auto"]
Optimizer = Literal["sgd", "adam"]

class Policy(torch.nn.Module):

    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 device: Device = "auto",
                 hidden_layers: int|Sequence[int] = 2) -> None:
        super().__init__()

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._net = torch.nn.Sequential()

        if isinstance(hidden_layers, int):
            for i in range(hidden_layers):
                self._net += torch.nn.Sequential(
                    torch.nn.Linear(1,1, device=self.device),
                    torch.nn.ReLU()
                )
        elif isinstance(hidden_layers, Sequence):
            for i in range(1, len(hidden_layers)):
                self._net += torch.nn.Sequential(
                    torch.nn.Linear(hidden_layers[i-1], hidden_layers[i], device=self.device),
                    torch.nn.ReLU()
                )
        else:
            raise TypeError(f"Argument for layers has incompatible type: {type(hidden_layers)}")
        
    def forward(self, tensor: Tensor) -> Tensor:
        return self._net.forward(tensor)
    
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
