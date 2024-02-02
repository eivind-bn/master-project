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
        self._gradients: Tensor|None = None
    
    def gradients(self, to_scalars: Callable[[Tensor],Tensor]) -> Tensor:
        if self._gradients is None:
            requires_grad = self._input.requires_grad
            self._input.requires_grad = True
            outputs = to_scalars(self._output)
            self._gradients = torch.autograd.grad(
                outputs=outputs,
                inputs=self._input,
                grad_outputs=torch.ones_like(outputs)
            )[0]
            self._input.requires_grad = requires_grad
        
        return self._gradients
    
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