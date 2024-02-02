from typing import *
from abc import ABC, abstractmethod
from typing import Any, Callable
from torch import Tensor
from torch.nn import Parameter, Module
from tqdm import tqdm
from torch.optim import Optimizer as TorchOptimizer
from .loss import LossType, LossModule, LossFunction, LossName, LossSelector

import torch
import inspect

T = TypeVar("T", contravariant=True)
P = TypeVar("P", covariant=True)

class ParamCall(Protocol[T,P]):
    def __call__(self, step: int, params: T) -> P: ...

Param = Union[P,ParamCall[T,P]]

class Optimizer(ABC):

    def __init__(self, 
                 optimizer_type:    Type[TorchOptimizer], 
                 trainable_params:  Iterable[Parameter], 
                 forward:           Callable[[Tensor],Tensor],
                 **hyperparameters: Any) -> None:
        
        self._optimizer_type = optimizer_type
        self._optimizer: TorchOptimizer|None = None
        self._f_params = trainable_params
        self._f = forward

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

            self._optimizer = self._optimizer_type(self._f_params, **params)
        else:
            for param_group in self._optimizer.param_groups:

                for key,value in self._param_values.items():
                    param_group[key] = value

                for key,callable_value in self._param_callables.items():
                    param_group[key] = callable_value(step, param_group)

        return self._optimizer

    def fit(self, 
            X:              Tensor, 
            Y:              Tensor, 
            steps:          int, 
            batch_size:     int,
            loss_criterion: LossName|LossSelector|LossFunction|LossModule,
            verbose:        bool = False) -> None:
        assert X.shape[0] == Y.shape[0] and 0 < batch_size <= X.shape[0]

        if isinstance(loss_criterion, LossModule):
            loss_function = loss_criterion
        elif isinstance(loss_criterion, str):
            loss_function = LossModule.get(loss_criterion)()
        elif callable(loss_criterion):
            args = inspect.getfullargspec(loss_criterion).args
            match len(args):
                case 1:
                    loss_function = cast(LossSelector, loss_criterion)(LossModule)
                case 2:
                    loss_function = LossModule(cast(LossFunction, loss_criterion))
                case _:
                    raise TypeError(f"Callable must accept either 1 or 2 arguments, but accepts {len(args)}")
        else:
            raise TypeError(f"Incompatible loss criterion: {type(loss_criterion)}")

        
        def mini_batch() -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
            return X[idx], Y[idx]

        with tqdm(total=steps, desc="Step:", disable=not verbose) as bar:
            for step in range(steps):
                optimizer = self.get_optimizer(step)
                optimizer.zero_grad()
                x,y = mini_batch()
                y_hat = self._f(x)
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
                 trainable_params:  Iterable[Parameter], 
                 forward:         Callable[[Tensor], Tensor], 
                 **params:  Unpack[Params]) -> None:
        super().__init__(torch.optim.SGD, trainable_params, forward, **params)

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
                 trainable_params:  Iterable[Parameter], 
                 forward:         Callable[[Tensor], Tensor], 
                 **params:  Unpack[Params]) -> None:
        super().__init__(torch.optim.Adam, trainable_params, forward, **params)