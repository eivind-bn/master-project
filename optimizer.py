# %%

from typing import *
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Type
from torch import Tensor
from torch.nn import Parameter
from tqdm import tqdm
from torch.optim import Optimizer as TorchOptimizer

import torch.nn.functional as F
import torch


T = TypeVar("T", bound=TorchOptimizer)
P = TypeVar("P")

Param = P|Callable[[],P]

class Optimizer(Generic[T]):

    def __init__(self, 
                 optimizer_type:    Type[T], 
                 f_params:          Iterable[Parameter], 
                 f:                 Callable[[Tensor],Tensor],
                 **params:          Any) -> None:
        
        self._optimizer_type = optimizer_type
        self._optimizer: T|None = None
        self._f_params = f_params
        self._f = f

        self._param_values: Dict[str,Any] = {}
        self._param_callables: Dict[str,Callable[[],Any]] = {}

        for name,param in params.items():
            if callable(param):
                self._param_callables[name] = param
            else:
                self._param_values[name] = param

    def update_params(self) -> None:
        for param_group in self._optimizer.param_groups:

            for key,value in self._param_values.items():
                param_group[key] = value

            for key,callable_value in self._param_callables.items():
                param_group[key] = callable_value()

    def get_optimizer(self) -> T:
        if self._optimizer is None:
            params = self._param_values.copy()

            for key,callable_value in self._param_callables.items():
                params[key] = callable_value()

            self._optimizer = self._optimizer_type(self._f_params, **params)
        else:
            self.update_params()

        return self._optimizer

    def fit(self, 
            X:              Tensor, 
            Y:              Tensor, 
            steps:          int, 
            batch_size:     int,
            loss_criterion: Callable[[Tensor,Tensor],Tensor],
            verbose:        bool = False) -> None: # type: ignore[misc]
        assert X.shape[0] == Y.shape[0] and 0 < batch_size <= X.shape[0]

        def mini_batch() -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
            return X[idx], Y[idx]

        with tqdm(total=steps, desc="Step:", disable=not verbose) as bar:
            for _ in range(steps):
                optimizer = self.get_optimizer()
                optimizer.zero_grad()
                x,y = mini_batch()
                y_hat = self._f(x)
                loss = loss_criterion(y_hat, y)
                optimizer.step()
                bar.set_description(f"Loss: {loss:.6f}")
                bar.update()

class SGD(Optimizer[torch.optim.SGD]):
    class Params(TypedDict, total=False):
        lr:             Required[Param[float]]
        momentum:       Param[float]
        dampening:      Param[float]
        weight_decay:   Param[float]
        nesterov:       Param[bool]

    def __init__(self, 
                 f_params:  Iterable[Parameter], 
                 f:         Callable[[Tensor], Tensor], 
                 **params:  Unpack[Params]) -> None:
        super().__init__(torch.optim.SGD, f_params, f, **params)

class Adam(Optimizer[torch.optim.Adam]):
    class Params(TypedDict, total=False):
        lr:             float|Tensor
        betas:          Tuple[float,float] 
        eps:            float
        weight_decay:   float
        amsgrad:        bool
        foreach:        bool|None
        maximize:       bool
        capturable:     bool
        differentiable: bool
        fused:          bool|None

    def __init__(self, 
                 f_params:  Iterable[Parameter], 
                 f:         Callable[[Tensor], Tensor], 
                 **params:  Unpack[Params]) -> None:
        super().__init__(torch.optim.Adam, f_params, f, **params)


# %%
f = torch.nn.Linear(10,100)
y_hat: Tensor = f(torch.full((3,10),10,dtype=torch.float32))


sgd = Adam(
    f_params=f.parameters(),
    f=lambda t: f(t),
    lr=0.1,
    nesterov=""
)

sgd.fit(
    torch.full((3,10), 100, dtype=torch.float32),
    torch.full((3,10), 100, dtype=torch.float32),
    100,
    2, 
    loss_criterion=lambda t1,t2: print("Hi")
    )

# %%
param = {"lr": 0.2}
s = torch.optim.SGD(f.parameters(), **param)
s.register_step_pre_hook(lambda *k: print("Hrllo"))
s.zero_grad()
y_hat.sum().backward(retain_graph=True)
s.step()
# %%