from typing import *
from abc import ABC, abstractmethod
from typing import Any, Callable
from torch import Tensor
from tqdm import tqdm
from torch.optim import Optimizer as TorchOptimizer
from .policy import Policy
from .loss import LossFunction

import torch.nn.functional as F
import torch

P = ParamSpec("P")
T = TypeVar("T", bound=TorchOptimizer)

class OptimizerType(Generic[P]):

    def __init__(self, constructor: Callable[P,T]) -> None:
        super().__init__()
        self._constructor = constructor

    def __call__(self, f: Callable[[Tensor],Tensor], *args: P.args, **kwargs: P.kwargs) -> "Optimizer":
        return Optimizer(f, self._constructor(*args, **kwargs))

class Optimizer:

    def __init__(self, f: Callable[[Tensor],Tensor], optimizer: TorchOptimizer) -> None:
        self._f = f
        self._optimizer = optimizer

    def fit(self, 
            X:              Tensor, 
            Y:              Tensor, 
            steps:          int, 
            batch_size:     int,
            loss_criterion: LossFunction,
            verbose:        bool = False) -> None: # type: ignore[misc]
        assert X.shape[0] == Y.shape[0] and 0 < batch_size <= X.shape[0]

        def mini_batch() -> Tuple[Tensor,Tensor]:
            idx = torch.randperm(X.shape[0], device=X.device)[:batch_size]
            return X[idx], Y[idx]

        with tqdm(total=steps, desc="Step:", disable=not verbose) as bar:
            for _ in range(steps):
                self._optimizer.zero_grad()
                x,y = mini_batch()
                y_hat = self._f(x)
                loss = loss_criterion(y_hat, y)
                self._optimizer.step()
                bar.set_description(f"Loss: {loss:.6f}")
                bar.update()


    class FitParams(TypedDict):
        X:              Tensor
        Y:              Tensor
        steps:          int
        batch_size:     int
        loss_criterion: LossFunction
        verbose:        NotRequired[bool]

    def mse_fit(self, *,
                **params: Unpack[FitParams],
                size_average: bool|None = None,
                reduce: bool | None = None,
                reduction: str = "mean"):
        self.fit(
            loss_criterion=lambda y_hat,y: F.mse_loss(y_hat, y, size_average, reduce, reduction),
            **params
        )


