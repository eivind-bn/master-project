from typing import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from torch import Tensor, device
from torch.nn import Sequential, Parameter, Module
from .feed_forward import FeedForward

import torch
import copy
import math

Device = Literal["cpu", "cuda", "cuda:0", "cuda:1", "auto"]
Activation = Literal["relu", "sigmoid", "tanh"]

P = TypeVar("P", bound="Policy")

@dataclass(frozen=True)
class Policy(ABC):
    input_dim:      Tuple[int,...]
    output_dim:     Tuple[int,...]
    device:         Device
    network:        Sequential

    @abstractmethod
    def loss_function(self) -> Callable[[Tensor,Tensor],Tensor]:
        pass

    def predict(self, 
                x:              Tensor|FeedForward, 
                move_to_device: bool = False) -> FeedForward:
        
        if isinstance(x, FeedForward):
            x = x._output

        if move_to_device:
            x = x.to(device=self.device)

        if len(x.shape) == len(self.input_dim) + 1:
            return FeedForward(
                input=x,
                network=self.network,
                loss_function=self.loss_function()
            )
        else:
            return FeedForward(
                input=x,
                network=self.network,
                loss_function=self.loss_function()
            )
            
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
        
    @classmethod
    def new(cls,
            input_dim:      int|Sequence[int],
            output_dim:     int|Sequence[int],
            hidden_layers:  int|Sequence[int] = 2,
            activation:     Activation|None = "relu",
            device:         Device = "auto") -> Self:
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        activations: Dict[Activation,Type[Module]] = {
            "relu": torch.nn.ReLU,
            "sigmoid": torch.nn.Sigmoid,
            "tanh": torch.nn.Tanh
        }

        network = torch.nn.Sequential()

        if isinstance(input_dim, Sequence):
            input_dim = tuple(input_dim)
        else:
            input_dim = (input_dim,)

        if isinstance(output_dim, Sequence):        
            output_dim = tuple(output_dim)
        else:
            output_dim = (output_dim,)

        I = math.prod(input_dim)
        O = math.prod(output_dim)

        if isinstance(hidden_layers, Sequence):
            layers = [I] + list(hidden_layers) + [O]
        else:
            N = hidden_layers + 1
            layers = [I] + [int(I - ((I - O)*n)/(N)) for n in range(1, N)] + [O]
        
        for i in range(1, len(layers)-1):
            network.append(torch.nn.Linear(layers[i-1], layers[i], device=device))
            if activation is not None:
                network.append(activations[activation]())

        network.append(torch.nn.Linear(layers[-2], layers[-1], device=device))

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            network=network,
        )
    
    def __add__(self, other: P) -> P:
        assert self.output_dim == other.input_dim
        return other.__class__(
            input_dim=self.input_dim,
            output_dim=other.output_dim,
            device=self.device,
            network=self.network + other.network,
        )
    
    def __call__(self, x: Tensor|FeedForward) -> FeedForward:
        return self.predict(x)
    
    def __repr__(self) -> str:
        return str(self.network)

class ContinuousPolicy(Policy):

    def loss_function(self) -> Callable[[Tensor, Tensor], Tensor]:
        return torch.nn.functional.mse_loss
    
class DiscretePolicy(Policy):

    def loss_function(self) -> Callable[[Tensor, Tensor], Tensor]:
        return torch.nn.functional.cross_entropy