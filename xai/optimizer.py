from typing import *
from abc import ABC
from typing import Any, Callable, Type
from torch import Tensor
from torch.nn import Parameter
from numpy import ndarray
from tqdm import tqdm
from torch.optim import Optimizer as TorchOptimizer
from .loss import LossModule, Loss
from .feed_forward import FeedForward
from .stats import TrainStats
from .reflist import RefList


import torch

if TYPE_CHECKING:
    from .network import Network

T = TypeVar("T", contravariant=True)
P = TypeVar("P", covariant=True)

class ParamCall(Protocol[T,P]):
    def __call__(self, step: int, params: T) -> P: ...

Param = Union[P,ParamCall[T,P]]

class Optimizer(ABC):

    def __init__(self, 
                 optimizer_type:    Type[TorchOptimizer], 
                 network:            "Network",
                 **hyperparameters: Any) -> None:
        
        self._optimizer_type = optimizer_type
        self._optimizer: TorchOptimizer|None = None
        self._network = network

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

            self._optimizer = self._optimizer_type(self._network.parameters(), **params)
        else:
            for param_group in self._optimizer.param_groups:

                for key,value in self._param_values.items():
                    param_group[key] = value

                for key,callable_value in self._param_callables.items():
                    param_group[key] = callable_value(step, param_group)

        return self._optimizer

    def fit(self, 
            X:              Tensor|ndarray, 
            Y:              Tensor|ndarray, 
            epochs:         int, 
            batch_size:     int,
            loss_criterion: Loss,
            verbose:        bool = False,
            info:           str|None = None) -> TrainStats:
        
        assert len(X) == len(Y), f"Length of X={len(X)} differs from length of Y={len(Y)}"
        assert 0 < batch_size <= len(X), f"{batch_size=} is not between 0 and {len(X)}"

        def prepare_tensor(array: Tensor|ndarray) -> Tensor:
            if isinstance(array, ndarray):
                array = torch.from_numpy(X)
            else:
                array = array.detach()

            return array.to(device=self._network.device, dtype=torch.float32)

        X = prepare_tensor(X)
        Y = prepare_tensor(Y)

        loss_function = LossModule.get(loss_criterion)
        losses: List[float] = []
        
        def mini_batch() -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(len(X))[:batch_size]
            return X[idx], Y[idx]

        with tqdm(total=epochs, desc="Step:", disable=not verbose) as bar:
            for epoch in range(epochs):
                optimizer = self.get_optimizer(epoch)
                optimizer.zero_grad()
                x,y = mini_batch()
                y_hat = self._network(x).output()
                loss: Tensor = loss_function(y_hat, y)
                losses.append(float(loss.item()))
                loss.backward()
                optimizer.step()
                bar.set_description(f"Loss: {loss:.6f}")
                bar.update()

        return TrainStats(
            batch_size=batch_size,
            losses=tuple(losses),
            info=info
        )

class SGD(Optimizer):
    class Params(TypedDict, total=False):
        lr:             Required[Param[float,"SGD.Params"]]
        momentum:       Param[float,"SGD.Params"]
        dampening:      Param[float,"SGD.Params"]
        weight_decay:   Param[float,"SGD.Params"]
        nesterov:       Param[bool,"SGD.Params"]

    def __init__(self, 
                 network:            "Network", 
                 **params:          Unpack[Params]) -> None:
        super().__init__(torch.optim.SGD, network, **params)

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
                 network:        "Network", 
                 **params:      Unpack[Params]) -> None:
        super().__init__(torch.optim.Adam, network, **params)

class RMSprop(Optimizer):
    class Params(TypedDict, total=False):
        lr:             Param[float, "RMSprop.Params"]
        alpha:          Param[float, "RMSprop.Params"]
        eps:            Param[float, "RMSprop.Params"]
        weight_decay:   Param[float, "RMSprop.Params"]
        momentum:       Param[float, "RMSprop.Params"]
        centered:       Param[bool, "RMSprop.Params"]

    def __init__(self, 
                 network:        "Network", 
                 **params:      Unpack[Params]) -> None:
        super().__init__(torch.optim.RMSprop, network, **params)
