from typing import *
from dataclasses import dataclass
from numpy import ndarray
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Parameter, Module, Sequential
from xai import Device, get_device
from .optimizer import SGD, Adam, RMSprop
from .activation import Activation, ActivationModule
import torch
import dill # type: ignore
import math
from numpy.typing import NDArray

Ints: TypeAlias = Tuple[int,...]
Sx = TypeVar("Sx", bound=Ints)
Sy = TypeVar("Sy", bound=Ints)
Sz = TypeVar("Sz", bound=Ints)

class Network(Generic[Sx,Sy]):
    @dataclass
    class FeedForward:
        _input:      Tensor
        _output:    Callable[[], Tensor]|Tensor

        @property
        def input(self) -> Tensor:
            return self._input

        @property
        def output(self) -> Tensor:
            if not isinstance(self._output, Tensor):
                self._output = self._output()

            return self._output

    def __init__(self, 
                 device:        Device,
                 input_shape:        Sx,
                 output_shape:       Sy) -> None:
        
        self._logits: Tuple[Callable[[Tensor],Tensor],...] = tuple()
        self._device = get_device(device)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._items = 1
        
        for dim in self._output_shape:
            if dim > 0:
                self._items *= dim
            else:
                raise ValueError(f"Shape axis must be positive, not {dim}")
            
    @property
    def input_shape(self) -> Sx:
        return self._input_shape
    
    @property
    def output_shape(self) -> Sy:
        return self._output_shape
            
    @staticmethod
    def new(device: Device, input_shape: Sx) -> "Network[Sx,Sx]":
        return Network(
            device=device,
            input_shape=input_shape,
            output_shape=input_shape
        )
            
    @staticmethod
    def dense(input_dim:          Sx,
              output_dim:         Sy,
              hidden_layers:      int|Sequence[int] = 2,
              hidden_activation:  Activation|None = "ReLU",
              output_activation:  Activation|None = None,
              device:             Device = "auto") -> "Network[Sx,Sy]":

        network = Network.new(device, input_dim).flatten()

        I = math.prod(input_dim)
        O = math.prod(output_dim)

        if isinstance(hidden_layers, Sequence):
            layers = list(hidden_layers) + [O]
        else:
            N = hidden_layers + 1
            layers = [int(I - ((I - O)*n)/(N)) for n in range(1, N)] + [O]

        for layer in layers[:-1]:
            network = network.linear(layer)
            if hidden_activation:
                network = network.activation(hidden_activation)

        network = network.linear(layers[-1])

        if output_activation:
            network = network.activation(output_activation)
            
        return network.reshape(output_dim)

    def reshape(self, shape: Sz) -> "Network[Sx,Sz]":
        items = 1
        new_shape = list(shape)
        unknown_dims: List[int] = []

        for i,d in enumerate(new_shape):
            if d != -1:
                items *= d
            else:
                unknown_dims.append(i)

        if len(unknown_dims) > 1:
            raise ValueError(f"Too many unknown dims: {len(unknown_dims)}. Max one permitted.")
        
        for unknown_dim in unknown_dims:
            if self._items % items == 0:
                new_shape[unknown_dim] = self._items // items
            else:
                raise ValueError(f"Cannot infer new dim")
            
        shape = cast(Sz, tuple(new_shape))

        network = self._appended(lambda tensor: tensor.reshape(shape), shape)
        assert self._items == network._items, f"New rank: {network._items} differs from current rank: {self._items}"
        return network
    
    def flatten(self) -> "Network[Sx,Tuple[int]]":
        return self.reshape((-1,))

    def linear(self: "Network[Sx,Tuple[int]]", dim: int) -> "Network[Sx,Tuple[int]]":
        assert len(self._output_shape) == 1, f"Output dim must be flattened, but is: {self._output_shape}"
        return self._appended(torch.nn.Linear(self._output_shape[0], dim, device=self._device), (dim,))
    
    def relu(self) -> "Network[Sx,Sy]":
        return self.activation("ReLU")
    
    def sigmoid(self) -> "Network[Sx,Sy]":
        return self.activation("Sigmoid")
    
    def activation(self, name: Activation) -> "Network[Sx,Sy]":
        return self._appended(ActivationModule.get(name), self._output_shape)
    
    def modules(self) -> Iterator[Module]:
        for layer in self._logits:
            if isinstance(layer, Module):
                yield layer
    
    def parameters(self) -> Iterator[Parameter]:
        for module in self.modules():
            for param in module.parameters():
                yield param

    def _to_tensor(self, array: NDArray[Any]|Tensor) -> Tensor:
        if isinstance(array, ndarray):
            array = torch.from_numpy(array)

        return array.to(dtype=torch.float32, device=self._device)
        
    def __call__(self, array: NDArray[Any]|Tensor) -> FeedForward:
        input: Tensor = self._to_tensor(array)
        input_shape = tuple(input.shape)

        def forward(tensor: Tensor) -> Tensor:
            for layer in self._logits:
                tensor = layer(tensor)
            return tensor

        if input_shape[1:] == self._input_shape:
            return self.FeedForward(
                _input=input,
                _output=lambda: forward(input)
            )
        elif input_shape == self._input_shape:
            input = input.unsqueeze(0)
            return self.FeedForward(
                _input=input,
                _output=lambda: forward(input).squeeze(0)
            )
        else:
            raise ValueError(f"Incorrect input-shape: {input_shape}, expected: {self._input_shape}")

    def __add__(self, other: "Network[Sy,Sz]") -> "Network[Sx,Sz]":
        if self._output_shape != other._input_shape:
            raise ValueError(f"Incompatible operand shape: {self._output_shape} != {other._input_shape}")
        
        return self._extended(other._logits, other._output_shape)
    
    def _extended(self, 
                  logits:       Tuple[Callable[[Tensor],Tensor],...], 
                  new_shape:    Sz) -> "Network[Sx,Sz]":
        network = Network(
            device=self._device,
            input_shape=self._input_shape,
            output_shape=new_shape
        )
        network._logits = self._logits + logits
        return network
    
    def _appended(self,
                  f:            Callable[[Tensor],Tensor],
                  new_shape:   Sz) -> "Network[Sx,Sz]":
        return self._extended((f,), new_shape)

    def save(self, path: str) -> None:
        with open(path, "w+b") as file:
            dill.dump(self, file)

    def __repr__(self) -> str:
        return str(Sequential(*self.modules()))
    
    @overload
    @classmethod
    def load(cls, 
             path: str, 
             *, 
             input_shape: Sx, 
             output_shape: Sy) -> "Network[Sx,Sy]": ...

    @overload
    @classmethod
    def load(cls, 
             path: str, 
             *, 
             input_shape: Sx) -> "Network[Sx, Ints]": ...

    @overload
    @classmethod
    def load(cls, 
             path: str, 
             *, 
             output_shape: Sy) -> "Network[Ints, Sy]": ...

    @overload
    @classmethod
    def load(cls, 
             path: str) -> "Network[Ints, Ints]": ...

    @classmethod
    def load(cls, 
             path: str, 
             *, 
             input_shape: Sx|None = None, 
             output_shape: Sy|None = None) -> "Network":
        with open(path, "rb") as file:
            network: Network = dill.load(file)

        if isinstance(network, cls):
            if input_shape and input_shape != network._input_shape:
                raise TypeError(f"Unpickled object has incorrect input-shape: {input_shape}, should be: {network._input_shape}")  
            
            if output_shape and output_shape != network._output_shape:
                raise TypeError(f"Unpickled object has incorrect output-shape: {output_shape}, should be: {network._output_shape}")  

            return cast(Network[Any,Any], network)
        else:
            raise TypeError(f"Unpickled object is of incorrect type: {type(network)}")