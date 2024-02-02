from typing import *
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32

import torch
import matplotlib.pyplot as plt

class FeedForward:

    def __init__(self, 
                 input:     Tensor,
                 output:    Tensor) -> None:
        
        self._input = input
        self._output = output

    def derivatives(self, to_scalars: Callable[[Tensor],Tensor], max_order: int|None = 1) -> Iterator[Tensor]:
        derivative = self._output
        
        def next_derivative():
            nonlocal derivative
            while True:
                yield derivative.detach().requires_grad_(False)
                wrt = to_scalars(derivative)
                self._input.requires_grad = True
                derivative = torch.autograd.grad(
                    outputs=wrt,
                    inputs=self._input,
                    grad_outputs=torch.ones_like(wrt),
                    create_graph=True,
                    retain_graph=True
                )[0]
                self._input.requires_grad = False

        iterator = next_derivative()

        if max_order is None:
            while True:
                yield next(iterator)
        else:
            for _ in range(max_order+1):
                yield next(iterator)

    def derivative(self, to_scalars: Callable[[Tensor],Tensor], order: int = 1) -> Tensor:
        return tuple(self.derivatives(to_scalars, order))[order]

    def gradients(self, to_scalars: Callable[[Tensor],Tensor]) -> Tensor:
        return self.derivative(to_scalars, order=1)
    
    def tensor(self) -> Tensor:
        return self._output
    
    def numpy(self) -> NDArray[float32]:
        return cast(NDArray[float32], self._output
                    .detach()
                    .cpu()
                    .numpy())
    
    def show(self, colormap: str = "gray") -> None:
        plt.imshow(self.numpy(), cmap=colormap)
        plt.show()
    
    def __repr__(self) -> str:
        return str(self._output)