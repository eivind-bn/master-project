from typing import *
from abc import ABC
from typing import Any, Callable
from torch import Tensor
from torch.nn import Parameter
from numpy import ndarray
from tqdm import tqdm
from torch.optim import Optimizer as TorchOptimizer
from .loss import LossModule, Loss
from .feed_forward import FeedForward

import torch

if TYPE_CHECKING:
    from .policy import Policy

T = TypeVar("T", contravariant=True)
P = TypeVar("P", covariant=True)

class ParamCall(Protocol[T,P]):
    def __call__(self, step: int, params: T) -> P: ...

Param = Union[P,ParamCall[T,P]]

class Optimizer(ABC):

    def __init__(self, 
                 optimizer_type:    Type[TorchOptimizer], 
                 policy:            "Policy",
                 **hyperparameters: Any) -> None:
        
        self._optimizer_type = optimizer_type
        self._optimizer: TorchOptimizer|None = None
        self._policy = policy

        self._param_values: Dict[str,Any] = {}
        self._param_callables: Dict[str,ParamCall[Any,Any]] = {}

        for name,param in hyperparameters.items():
            if callable(param):
                self._param_callables[name] = param
            else:
                self._param_values[name] = param

    def get_optimizer(self, step: int) -> TorchOptimizer:
        if self._optimizer is None:
            params = self._param_values.copy()

            for key,callable_value in self._param_callables.items():
                params[key] = callable_value(step, params)

            self._optimizer = self._optimizer_type(self._policy.network.parameters(), **params)
        else:
            for param_group in self._optimizer.param_groups:

                for key,value in self._param_values.items():
                    param_group[key] = value

                for key,callable_value in self._param_callables.items():
                    param_group[key] = callable_value(step, param_group)

        return self._optimizer

    def fit(self, 
            X:              Tensor|ndarray|FeedForward, 
            Y:              Tensor, 
            steps:          int, 
            batch_size:     int,
            loss_criterion: Loss,
            verbose:        bool = False) -> None:
        
        if isinstance(X, FeedForward):
            X = X.tensor()
        elif isinstance(X, ndarray):
            X = torch.from_numpy(X).to(device=self._policy.device, dtype=torch.float32)
            
        assert X.shape[0] == Y.shape[0] and 0 < batch_size <= X.shape[0]

        loss_function = LossModule.get(loss_criterion)

        X = X.detach()
        Y = Y.detach()
        
        def mini_batch() -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
            return X[idx], Y[idx]

        with tqdm(total=steps, desc="Step:", disable=not verbose) as bar:
            for step in range(steps):
                optimizer = self.get_optimizer(step)
                optimizer.zero_grad()
                x,y = mini_batch()
                y_hat = self._policy(x).tensor()
                loss: Tensor = loss_function(y_hat, y)
                loss.backward()
                optimizer.step()
                bar.set_description(f"Loss: {loss:.6f}")
                bar.update()

class SGD(Optimizer):
    class Params(TypedDict, total=False):
        lr:             Required[Param[float,"SGD.Params"]]
        momentum:       Param[float,"SGD.Params"]
        dampening:      Param[float,"SGD.Params"]
        weight_decay:   Param[float,"SGD.Params"]
        nesterov:       Param[bool,"SGD.Params"]

    def __init__(self, 
                 policy:            "Policy", 
                 **params:          Unpack[Params]) -> None:
        super().__init__(torch.optim.SGD, policy, **params)

class Adam(Optimizer):
    class Params(TypedDict, total=False):
        lr:             Param[float|Tensor,"Adam.Params"]
        betas:          Param[Tuple[float,float],"Adam.Params"]
        eps:            Param[float,"Adam.Params"]
        weight_decay:   Param[float,"Adam.Params"]
        amsgrad:        Param[bool,"Adam.Params"]
        foreach:        Param[bool|None,"Adam.Params"]
        maximize:       Param[bool,"Adam.Params"]
        capturable:     Param[bool,"Adam.Params"]
        differentiable: Param[bool,"Adam.Params"]
        fused:          Param[bool|None,"Adam.Params"]

    def __init__(self, 
                 policy:            "Policy", 
                 **params:          Unpack[Params]) -> None:
        super().__init__(torch.optim.Adam, policy, **params)