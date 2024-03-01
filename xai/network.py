from typing import *
from numpy import ndarray
from numpy.typing import NDArray
from torch import Tensor
from . import Device

import torch

Shape = TypeVarTuple("Shape")
NewShape = TypeVarTuple("NewShape")
Ints = Tuple[int,...]

class Network(Generic[*Shape]):

    def __init__(self, 
                 device:    Device,
                 *shape:    *Shape) -> None:
        self._logits: Tuple[Callable[[Tensor],Tensor],...] = tuple()
        self._shape = shape
        self._device = device

    def _appended(self, 
                  f:            Callable[[Tensor],Tensor], 
                  *new_shape:    *NewShape) -> "Network[*NewShape]":
        network = Network(self._device, *new_shape)
        network._logits = network._logits + (f,)
        return network

    def reshape(self: "Network[*Ints]", *shape: *NewShape) -> "Network[*NewShape]":
        return self._appended(lambda tensor: tensor.reshape(cast(Tuple[int,...], shape)), *shape)

    def linear(self: "Network[int]", dim: int) -> "Network[int]":
        return self._appended(torch.nn.Linear(self._shape[0], dim), dim)
    
    def __call__(self, array: NDArray[Any]|Tensor) -> Tensor:
        if isinstance(array, ndarray):
            array = torch.from_numpy(array)

        array = array.to(dtype=torch.float32, device=self._device)

        if array.shape[1:] == self._shape:
            pass

n = Network("auto", 4).reshape(2,3)




