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
                 set_device:        bool,
                 **hyperparameters: Any) -> None:
        
        self._optimizer_type = optimizer_type
        self._optimizer: TorchOptimizer|None = None
        self._policy = policy
        self._set_device = set_device

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
            X:              Tensor|ndarray|FeedForward|Sequence[Tensor], 
            Y:              Tensor|ndarray|FeedForward|Sequence[Tensor], 
            epochs:         int, 
            batch_size:     int,
            loss_criterion: Loss,
            verbose:        bool = False,
            info:           str|None = None) -> TrainStats:
        
        def get_index_access_function(array: Tensor|ndarray|FeedForward|Sequence[Tensor]) -> Tuple[int, Callable[[Tensor],Tensor]]:
            if isinstance(array, FeedForward):
                tensor = array.tensor(True).to(device=self._policy.device, dtype=torch.float32)
                return tensor.shape[0], lambda indices: tensor[indices]
            elif isinstance(array, ndarray):
                tensor = torch.from_numpy(array).to(device=self._policy.device, dtype=torch.float32)
                return tensor.shape[0], lambda indices: tensor[indices]
            elif isinstance(array, Tensor):
                tensor = array.to(device=self._policy.device, dtype=torch.float32)
                return array.shape[0], lambda indices: array[indices]
            else:
                def access(indices: Tensor) -> Tensor:
                    indices_list: List[int] = indices.tolist()
                    return torch.stack([array[index] for index in indices_list]).to(device=self._policy.device, dtype=torch.float32)
                return len(array), access
        
        X_len, get_X = get_index_access_function(X)
        Y_len, get_Y = get_index_access_function(Y)
            
        assert X_len == Y_len, f"{X_len=} differs from {Y_len=}"
        assert 0 < batch_size <= X_len, f"{batch_size=} is not between 0 and {X_len}"

        loss_function = LossModule.get(loss_criterion)
        losses: List[float] = []
        
        def mini_batch() -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(X_len)[:batch_size]
            return get_X(idx), get_Y(idx)

        with tqdm(total=epochs, desc="Step:", disable=not verbose) as bar:
            for epoch in range(epochs):
                optimizer = self.get_optimizer(epoch)
                optimizer.zero_grad()
                x,y = mini_batch()
                y_hat = self._policy(x).tensor(False)
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
                 policy:            "Policy", 
                 set_device:        bool,
                 **params:          Unpack[Params]) -> None:
        super().__init__(torch.optim.SGD, policy, set_device, **params)

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
                 policy:        "Policy", 
                 set_device:    bool,
                 **params:      Unpack[Params]) -> None:
        super().__init__(torch.optim.Adam, policy, set_device, **params)

class RMSprop(Optimizer):
    class Params(TypedDict, total=False):
        lr:             Param[float, "RMSprop.Params"]
        alpha:          Param[float, "RMSprop.Params"]
        eps:            Param[float, "RMSprop.Params"]
        weight_decay:   Param[float, "RMSprop.Params"]
        momentum:       Param[float, "RMSprop.Params"]
        centered:       Param[bool, "RMSprop.Params"]

    def __init__(self, 
                 policy:        "Policy", 
                 set_device:    bool, 
                 **params:      Unpack[Params]) -> None:
        super().__init__(torch.optim.RMSprop, policy, set_device, **params)
