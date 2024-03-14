from . import *
from abc import ABC
from typing import Any, Callable, Type
from torch import Tensor
from numpy import ndarray
from tqdm import tqdm
from torch.optim import Optimizer as TorchOptimizer

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
            X_train:            Tensor|ndarray, 
            Y_train:            Tensor|ndarray, 
            epochs:             int, 
            batch_size:         int,
            loss_criterion:     Loss,
            X_val:              Tensor|ndarray|None = None,
            Y_val:              Tensor|ndarray|None = None,
            is_correct:         Callable[[Tensor,Tensor],Tensor]|None = None,
            early_stop_count:   int|None = None,
            verbose:            bool = False,
            info:               str|None = None) -> TrainStats:
        
        assert len(X_train) == len(Y_train), f"Length of X_train={len(X_train)} differs from length of Y_train={len(Y_train)}"
        assert 0 < batch_size <= len(X_train), f"{batch_size=} is not between 0 and {len(X_train)}"

        def prepare_tensor(array: Tensor|ndarray) -> Tensor:
            if isinstance(array, ndarray):
                array = torch.from_numpy(array)
            else:
                array = array.detach()

            return array.to(device=self._network.device)

        if X_val is not None and Y_val is not None:
            assert len(X_val) == len(Y_val), f"Length of X_val={len(X_val)} differs from length of Y_val={len(Y_val)}"
            assert 0 < batch_size <= len(X_val), f"{batch_size=} is not between 0 and {len(X_val)}"
            X_val = prepare_tensor(X_val)
            Y_val = prepare_tensor(Y_val)
            val_exists = True
        elif X_val is not None or Y_val is not None:
            raise ValueError(f"Both X_val and Y_val must be provided or none of them.")
        else:
            X_val = None
            Y_val = None
            val_exists = False

        best_validation_loss: float|None = None
        worse_cnt: int = 0

        if early_stop_count is not None:
            if val_exists:
                if early_stop_count < 0:
                    raise ValueError(f"Early stop count must be positive")   
            else:
                raise ValueError(f"Cannot early stop without validation data.")

            def early_stop(loss: float) -> bool:
                nonlocal best_validation_loss
                nonlocal worse_cnt
                if best_validation_loss is not None:
                    if loss < best_validation_loss:
                        best_validation_loss = loss
                        worse_cnt = 0
                    else:
                        worse_cnt += 1
                else:
                    best_validation_loss = loss

                return worse_cnt >= early_stop_count
        else:
            def early_stop(loss: float) -> bool:
                return False

        X_train = prepare_tensor(X_train)
        Y_train = prepare_tensor(Y_train)

        loss_function = LossModule.get(loss_criterion)
        
        def mini_batch(X: Tensor, Y: Tensor) -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(len(X))[:batch_size]
            return X[idx], Y[idx]

        with tqdm(total=epochs, desc="Step:", disable=not verbose) as bar:
            for epoch in range(epochs):
                optimizer = self.get_optimizer(epoch)
                optimizer.zero_grad()
                x_train,y_train = mini_batch(X_train, Y_train)
                y_hat_train = self._network(x_train).output()
                train_loss: Tensor = loss_function(y_hat_train, y_train)
                self._network.batch_sizes.append(batch_size)
                self._network.train_losses.append(float(train_loss.item()))
                train_loss.backward()
                optimizer.step()

                if X_val is not None and Y_val is not None:
                    x_val,y_val = mini_batch(X_val, Y_val)
                    y_hat_val = self._network(x_val).output()
                    val_loss = float(loss_function(y_hat_val, y_val).item())
                    self._network.validation_losses.append(val_loss)

                    if is_correct is not None:
                        correct_classification = is_correct(y_hat_val, y_val).reshape((batch_size,1))
                        if correct_classification.dtype != torch.bool:
                            raise ValueError(f"Expected accuracy tensor to consist of bool dtype, but found: {correct_classification.dtype}")
                        accuracy = float((correct_classification.count_nonzero() / correct_classification.nelement()).item())
                        self._network.accuracies.append(accuracy)

                    if early_stop(val_loss):
                        bar.set_description(f"Early stopping! Train-loss: {train_loss:.6f}, Val-loss: {val_loss:.6f}")
                        break
                    else:
                        bar.set_description(f"Train-loss: {train_loss:.6f}, Val-loss: {val_loss:.6f}")
                else:
                    bar.set_description(f"Loss: {train_loss:.6f}")

                bar.update()

        return self._network.train_stats(info)

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
