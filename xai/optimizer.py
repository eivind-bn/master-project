from typing import *
from abc import ABC, abstractmethod
from torch import Tensor
from tqdm import tqdm
from torch.optim import Optimizer as TorchOptimizer
from .policy import Policy

import torch

P = TypeVar("P", bound=Dict[str,Any])

class Optimizer(ABC, Generic[P]):

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._optimizer: TorchOptimizer|None = None

    @abstractmethod
    def get_optimizer(self, **params: Unpack[P]) -> TorchOptimizer: # type: ignore[misc]
        pass

    def fit(self, 
            X:          Tensor, 
            Y:          Tensor, 
            steps:      int, 
            batch_size: int,
            verbose:    bool = False, 
            **params:   Unpack[P]) -> None: # type: ignore[misc]
        assert X.shape[0] == Y.shape[0] and 0 < batch_size <= X.shape[0]

        optimizer = self.get_optimizer(**params)

        def mini_batch() -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(X.shape[0], device=self._policy.device)[:batch_size]
            return X[idx], Y[idx]

        with tqdm(total=steps, desc="Step:", disable=not verbose) as bar:
            for _ in range(steps):
                optimizer.zero_grad()
                x,y = mini_batch()
                loss = self._policy(x).loss(y)
                loss.backward()
                optimizer.step()
                bar.set_description(f"Loss: {loss:.6f}")
                bar.update()

class SGDParams(TypedDict, total=False):
    lr:             float
    weight_decay:   float
    momentum:       float
    dampening:      float
    nesterov:       bool

class SGD(Optimizer[SGDParams]):

          
    def get_optimizer(self, **params: Unpack[SGDParams]) -> TorchOptimizer: # type: ignore[override]
        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(self._policy.network.parameters(), **params)
        else:
            for param_group in self._optimizer.param_groups:
                param_group.update(params)
        return self._optimizer
    
    def fit(self,  # type: ignore[override]
            X:          Tensor, 
            Y:          Tensor, 
            steps:      int, 
            batch_size: int, 
            verbose:    bool = False, 
            **params:   Unpack[SGDParams]) -> None:
        return super().fit(X, Y, steps, batch_size, verbose=verbose, **params)

class AdamParams(TypedDict, total=False):
    lr:                float|Tensor
    weight_decay:      float
    betas:             Tuple[float,float]
    eps:               float
    amsgrad:           bool
    foreach:           bool|None
    maximize:          bool
    capturable:        bool
    differentiable:    bool
    fused:             bool|None  

class Adam(Optimizer):


    def get_optimizer(self, # type: ignore[override]
                      **params: Unpack[AdamParams]) -> TorchOptimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._policy.network.parameters(), **params)
        else:
            for param_group in self._optimizer.param_groups:
                param_group.update(params)

        return self._optimizer
    
    def fit(self, # type: ignore[override]
            X:          Tensor, 
            Y:          Tensor, 
            steps:      int, 
            batch_size: int, 
            verbose:    bool = False, 
            **params:   Unpack[AdamParams]) -> None:
        return super().fit(X, Y, steps, batch_size, verbose=verbose, **params)
    