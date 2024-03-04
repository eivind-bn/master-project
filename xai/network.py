from typing import *
from numpy import ndarray
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Parameter, Module
from xai import Device
from .optimizer import SGD, Adam, RMSprop

import torch
import math

Shape = TypeVarTuple("Shape")
NewShape = TypeVarTuple("NewShape")
Ints = Tuple[int,...]

class Network(Generic[*Shape]):

    def __init__(self, 
                 device:    Device,
                 *shape:    *Shape) -> None:
        
        self._typed_shape = shape
        self._logits: Tuple[Callable[[Tensor],Tensor],...] = tuple()
        self._rank = 1
        
        def shape_calc() -> Iterator[int]:
            for dim in shape:
                if isinstance(dim, int) and dim > 0:
                    self._rank *= dim
                    yield dim
                else:
                    raise TypeError(f"Shape axis must be int type, not {type(dim)}")

        self._input_shape = tuple(shape_calc())
        self._output_shape = self._input_shape
        self._device = device

    def reshape(self: "Network[*Ints]", *shape: *NewShape) -> "Network[*NewShape]":
        network = self._appended(lambda tensor: tensor.reshape(cast(Tuple[int,...], shape)), *shape)
        assert self._rank == network._rank, f"New rank: {network._rank} differs from current rank: {self._rank}"
        return network

    def linear(self: "Network[int]", dim: int) -> "Network[int]":
        assert len(self._output_shape) == 1, f"Output dim must be flattened, but is: {self._output_shape}"
        return self._appended(torch.nn.Linear(self._input_shape[0], dim, device=self._device), dim)
    
    def relu(self) -> "Network[*Shape]":

        return self._appended(torch.nn.ReLU(), *self._typed_shape)
    
    def sigmoid(self) -> "Network[*Shape]":
        return self._appended(torch.nn.Sigmoid(), *self._typed_shape)
    
    def modules(self) -> Iterator[Module]:
        for layer in self._logits:
            if isinstance(layer, Module):
                yield layer
    
    def parameters(self) -> Iterator[Parameter]:
        for module in self.modules():
            for param in module.parameters():
                yield param
        
    def __call__(self, array: NDArray[Any]|Tensor) -> Tensor:
        if isinstance(array, ndarray):
            array = torch.from_numpy(array)

        array = array.to(dtype=torch.float32, device=self._device)

        if array.shape[1:] == self._input_shape:
            Z = array
            for layer in self._logits:
                Z = layer(Z)
            return Z
        elif array.shape == self._input_shape:
            Z = array.unsqueeze(0)
            for layer in self._logits:
                Z = layer(Z)
            return Z.squeeze(0)
        else:
            raise ValueError(f"Incorrect input-shape: {array.shape}, expected: {self._input_shape}")

    def __add__(self, other: "Network[*NewShape]") -> "Network[*NewShape]":
        assert self._output_shape == other._input_shape
        return self._extended(other._logits, *other._typed_shape)
    
    def _extended(self, 
                  logits:       Tuple[Callable[[Tensor],Tensor],...], 
                  *new_shape:   *NewShape) -> "Network[*NewShape]":
        
        network = Network(self._device, *new_shape)
        network._logits = self._logits + logits
        network._input_shape = self._input_shape
        return network
    
    def _appended(self,
                  f:            Callable[[Tensor],Tensor],
                  *new_shape:   *NewShape) -> "Network[*NewShape]":
        return self._extended((f,), *new_shape)

    def save(self, path: str) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> Self:
        network: Self = torch.load(path)
        if isinstance(network, cls):
            return network
        else:
            raise TypeError(f"Unpickled object is of incorrect type: {type(network)}")