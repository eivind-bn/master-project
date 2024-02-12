from typing import *
from abc import ABC
from dataclasses import dataclass
from typing import Callable
from torch import Tensor, device
from torch.nn import Sequential, Parameter, Module
from numpy.typing import NDArray
from numpy import float32, ndarray
from .feed_forward import FeedForward
from .optimizer import SGD, Adam, RMSprop
from .fitness import NormalizedFitness
from .activation import Activation, ActivationModule

import torch
import copy
import math
import random

Device = Literal["cpu", "cuda", "cuda:0", "cuda:1", "auto"]

P = TypeVar("P", bound="Policy")

@dataclass(frozen=True)
class Policy(ABC):
    input_dim:      Tuple[int,...]
    output_dim:     Tuple[int,...]
    hidden_layers:  Tuple[int,...]
    device:         Device
    network:        Sequential
    normalize:      float|None
    set_device:     bool

    def sgd(self, **params: Unpack[SGD.Params]) -> SGD:
        return SGD(policy=self, set_device=self.set_device, **params)
    
    def adam(self, **params: Unpack[Adam.Params]) -> Adam:
        return Adam(policy=self, set_device=self.set_device, **params)
    
    def rms_prop(self, **params: Unpack[RMSprop.Params]) -> RMSprop:
        return RMSprop(policy=self, set_device=self.set_device, **params)
    
    def predict(self, 
                X:          Tensor|ndarray|FeedForward, 
                normalize:  float|None = None,
                detach:     bool = True) -> FeedForward:
        
        if isinstance(X, ndarray):
            X = (torch.from_numpy(X)
                 .to(device=self.device, dtype=torch.float32))
            
        elif isinstance(X, FeedForward):
            X = X.tensor(True)

            if self.set_device:
                X = X.to(device=self.device)

            if X.dtype != torch.float32:
                X = X.float()

        else:
            if detach:
                X = X.detach()  

            if self.set_device:
                X = X.to(device=self.device)

            if X.dtype != torch.float32:
                X = X.float()

        if normalize is not None:
            X /= normalize
        elif self.normalize is not None:
            X /= self.normalize

        X = X.requires_grad_(True)

        if X.shape == self.input_dim:
            Y: Tensor = self.network(X.flatten())
            Y = Y.reshape(self.output_dim)
        elif X.shape[-len(self.input_dim):] == self.input_dim:
            Y = self.network(X.flatten(start_dim=1))
            Y = Y.reshape((X.shape[0],) + self.output_dim)
        else:
            raise ValueError(f"Shape mismatch between {self.input_dim} and {X.shape}")
        
        X = X.requires_grad_(False)
        
        return FeedForward(
            input=X,
            output=Y
        )
    
    def mutate(self, volatility: float, rate: float|None = None) -> None:
        for params in self.network.parameters():
            with torch.no_grad():
                if rate is None:
                    params[:] = torch.normal(params, volatility)
                else:
                    rands = torch.rand(params.shape, device=params.device)
                    params[:] = torch.where(rands < rate, torch.normal(params, volatility), params)

    @staticmethod
    def crossover(policies: Iterable["Policy"], fitnesses: Iterable[NormalizedFitness|None]) -> "Policy":
        policies = tuple(policies)
        ranks = tuple(random.random() if fitness is None else fitness.rank() for fitness in fitnesses)
        rank_sum = sum(ranks)
        rank_portions = tuple(rank/rank_sum for rank in ranks)

        new_policy: Policy|None = None

        for policy,rank_portion in zip(policies,rank_portions, strict=True):
            with torch.no_grad():
                if new_policy is None:
                    new_policy = policy.copy()
                    for param in new_policy.network.parameters():
                        param[:] = 1.0

                for param,new_param in zip(policy.network.parameters(), new_policy.network.parameters()):
                    new_param[:] = rank_portion*param

        return new_policy
            
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
            input_dim:          int|Sequence[int],
            output_dim:         int|Sequence[int],
            hidden_layers:      int|Sequence[int] = 2,
            hidden_activation:  Activation|None = "ReLU",
            output_activation:  Activation|None = None,
            device:             Device = "auto",
            normalize:          float|None = None,
            set_device:         bool = True) -> Self:
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if hidden_activation is not None:
            hidden_activation = ActivationModule.get(hidden_activation)
        
        if output_activation is not None:
            output_activation = ActivationModule.get(output_activation)

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

        network = torch.nn.Sequential()
        
        for i in range(1, len(layers)-1):
            network.append(torch.nn.Linear(layers[i-1], layers[i], device=device))
            if hidden_activation is not None:
                network.append(hidden_activation.copy())

        network.append(torch.nn.Linear(layers[-2], layers[-1], device=device))

        if output_activation is not None:
            network.append(output_activation.copy())

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=tuple(layers[1:-1]),
            device=device,
            network=network,
            normalize=normalize,
            set_device=set_device
        )
    
    def __add__(self, other: P) -> P:
        assert self.output_dim == other.input_dim
        return other.__class__(
            input_dim=self.input_dim,
            output_dim=other.output_dim,
            hidden_layers=self.hidden_layers + other.hidden_layers,
            device=self.device,
            network=self.network + other.network,
            normalize=self.normalize,
            set_device=self.set_device
        )
    
    def __call__(self, 
                 X:             Tensor|NDArray[float32]|FeedForward, 
                 normalize:     float|None = None,
                 detach:        bool = True) -> FeedForward:
        return self.predict(
            X=X,
            normalize=normalize,
            detach=detach
        )
    
    def __repr__(self) -> str:
        return str(self.network)
