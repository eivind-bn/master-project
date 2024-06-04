from . import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray
from numpy.typing import NDArray
from torch import Tensor, device
from torch.nn import Parameter, Module, Sequential

import torch
import dill # type: ignore
import math

Array: TypeAlias = NDArray[Any]|Tensor
Ints: TypeAlias = Tuple[int,...]
T = TypeVar("T", bound=Explainer)
Sx = TypeVar("Sx", bound=Ints)
Sy = TypeVar("Sy", bound=Ints)
Sz = TypeVar("Sz", bound=Ints)

@dataclass
class FeedForward(Generic[Sx,Sy]):
    parent:     "Network[Sx,Sy]"
    input:      Lazy[Tensor]
    output:     Lazy[Tensor]
    
    def derivatives(self, to_scalars: Callable[[Tensor],Tensor], max_order: int|None = 1) -> Stream[Tensor]:  
        def next_derivative() -> Iterator[Tensor]:
            input = self.input()
            derivative = self.output()
            while True:
                yield derivative.detach().requires_grad_(False)
                wrt = to_scalars(derivative)
                derivative = torch.autograd.grad(
                    outputs=wrt,
                    inputs=input,
                    grad_outputs=torch.ones_like(wrt),
                    create_graph=True,
                    retain_graph=True
                )[0]

        if max_order is None:
            return Stream(next_derivative())
        else:
            return Stream(next_derivative()).take(max_order+1)
        
    def derivative(self, to_scalars: Callable[[Tensor],Tensor], order: int = 1) -> Tensor:
        return tuple(self.derivatives(to_scalars, order))[order]

    def gradients(self, to_scalars: Callable[[Tensor],Tensor]) -> Tensor:
        return self.derivative(to_scalars, order=1)
    
    def explain(self, 
                algorithm:  Type[Explainer]|Explainers,
                background: Array|None = None, 
                max_evals:  int|None = None) -> Explanation[Sy,Sx]:
        return self.parent.explainer(algorithm, background).explain(self.input(), max_evals=max_evals).item()
    
    def __call__(self) -> Tensor:
        return self.output()
        
    def __repr__(self) -> str:
        return str(self.output())
    
class Lambda(torch.nn.Module):

    def __init__(self, 
                 f:     Callable[[Tensor],Tensor], 
                 repr:  Callable[["Lambda"],str]) -> None:
        super().__init__()
        self._f = f
        self._repr = repr

    def forward(self, x: Tensor) -> Tensor:
        return self._f(x)
    
    def __repr__(self) -> str:
        if self._repr is None:
            return super().__repr__()
        else:
            return self._repr(self)
        
    def __getstate__(self) -> bytes:
        return dill.dumps(vars(self))
    
    def __setstate__(self, dump: bytes) -> None:
        self.__dict__ = dill.loads(dump)
        

class Network(Generic[Sx,Sy], Module):

    def __init__(self,
                 device:        Device,
                 input_shape:   Sx,
                 output_shape:  Sy,
                 logits:        Module|None = None) -> None:
        super().__init__()

        if logits is None:
            self._logits = Sequential()
        elif isinstance(logits, Sequential):
            self._logits = logits
        else:
            self._logits = Sequential(logits)

        self._device = get_device(device)
        self._logits = self._logits.to(device=self._device)
        
        self._input_shape = input_shape
        self._output_shape = output_shape

        self._items = 1
        self.train_history = TrainHistory()
        self.explainers: Dict[Type[Explainer],Tuple[Array,Explainer]] = {}
        
        for dim in self.output_shape:
            if dim > 0:
                self._items *= dim
            else:
                raise ValueError(f"Shape axis must be positive, not {dim}")
    
    @property
    def modules(self) -> Sequential:
        return self._logits
            
    @property
    def device(self) -> device:
        for param in self.parameters():
            self._device = param.device
            return self._device
        
        return self._device

    @property
    def input_shape(self) -> Sx:
        return self._input_shape
    
    @property
    def output_shape(self) -> Sy:
        return self._output_shape

    def exact_explainer(self, background: Array|None = None) -> ExactExplainer[Sy,Sx]:
        return cast(ExactExplainer, self.explainer(ExactExplainer, background))

    def permutation_explainer(self, background: Array|None = None) -> PermutationExplainer[Sy,Sx]:
        return cast(PermutationExplainer, self.explainer(PermutationExplainer, background))
    
    def kernel_explainer(self, background: Array|None = None) -> KernelExplainer[Sy,Sx]:
        return cast(KernelExplainer, self.explainer(KernelExplainer, background))
    
    def deep_explainer(self, background: Array|None = None) -> DeepExplainer[Sy,Sx]:
        return cast(DeepExplainer, self.explainer(DeepExplainer, background))

    def gradient_explainer(self, background: Array|None = None) -> GradientExplainer[Sy,Sx]:
        return cast(GradientExplainer, self.explainer(GradientExplainer, background))
    
    def explainer(self, 
                  algorithm: Type[Explainer]|Explainers, 
                  background: Array|None = None) -> Explainer[Sy,Sx]:
        if isinstance(algorithm, str):
            algorithm = {
                "exact": ExactExplainer,
                "permutation": PermutationExplainer,
                "deep": DeepExplainer,
                "kernel": KernelExplainer,
                "gradient": GradientExplainer
            }[algorithm]

        if background is not None:
            if algorithm in self.explainers:
                old_background, explainer = self.explainers[algorithm]
                if old_background is background:
                    return explainer
                
            self.explainers[algorithm] = (background, algorithm(self, background))

        return self.explainers[algorithm][1]
    
    def sgd(self, **params: Unpack[SGD.Params]) -> SGD:
        return SGD(network=self, **params)
    
    def adam(self, **params: Unpack[Adam.Params]) -> Adam:
        return Adam(network=self, **params)
    
    def rms_prop(self, **params: Unpack[RMSprop.Params]) -> RMSprop:
        return RMSprop(network=self, **params)
            
    @staticmethod
    def new(device: Device, input_shape: Sx) -> "Network[Sx,Sx]":
        return Network(
            device=device,
            input_shape=input_shape,
            output_shape=input_shape,
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

        network = self.appended(Network(
            device=self.device,
            input_shape=self.output_shape,
            output_shape=shape,
            logits=Sequential(Lambda(
                f=lambda tensor: tensor.reshape((-1,) + shape),
                repr=lambda _: f"Reshape({shape})"))
        ))
        assert self._items == network._items, f"New rank: {network._items} differs from current rank: {self._items}"
        return network
    
    def flatten(self) -> "Network[Sx,Tuple[int]]":
        return self.reshape((-1,))

    def linear(self: "Network[Sx,Tuple[int]]", dim: int) -> "Network[Sx,Tuple[int]]":
        assert len(self.output_shape) == 1, f"Output dim must be flattened, but is: {self.output_shape}"
        return self.appended(Network(
            device=self.device,
            input_shape=self.output_shape,
            output_shape=(dim,),
            logits=Sequential(torch.nn.Linear(self.output_shape[0], dim, device=self.device))
            ))

    def relu(self) -> "Network[Sx,Sy]":
        return self.activation("ReLU")
    
    def sigmoid(self) -> "Network[Sx,Sy]":
        return self.activation("Sigmoid")
    
    def activation(self, name: Activation) -> "Network[Sx,Sy]":
        return self.appended(Network(
            device=self.device,
            input_shape=self.output_shape,
            output_shape=self.output_shape,
            logits=Sequential(ActivationModule.get(name))
        ))

    def _to_tensor(self, array: Array) -> Tensor:
        if isinstance(array, ndarray):
            array = torch.from_numpy(array)

        return array.to(dtype=torch.float32, device=self.device)
        
    def forward(self, X: Array|Lazy[Array]|FeedForward[Ints,Sx]) -> FeedForward[Sx,Sy]:

        def forward(tensor: Tensor) -> Tensor:
            tensor = tensor.requires_grad_(True)
            if tensor.shape[1:] == self.input_shape:
                Z = tensor
                for layer in self.modules:
                    Z = layer(Z)
                return Z
            elif tensor.shape == self.input_shape:
                Z = tensor.unsqueeze(0)
                for layer in self.modules:
                    Z = layer(Z)
                return Z.squeeze(0)
            else:
                raise ValueError(f"Incorrect input-shape: {tuple(tensor.shape)}, expected: {self.input_shape}")
             
        if isinstance(X, FeedForward):
            return FeedForward(
                parent=self,
                input=X.output,
                output=X.output.map(forward)
            )
        else:
            input = Lazy[Array](lambda: X).map(self._to_tensor)
            return FeedForward(
                parent=self,
                input=input,
                output=input.map(forward)
            )
        
    def __call__(self, X: Array|Lazy[Array]|FeedForward[Ints,Sx]) -> FeedForward[Sx,Sy]:
        return cast(FeedForward[Sx,Sy], super().__call__(X))

    def __add__(self, other: "Network[Sy,Sz]") -> "Network[Sx,Sz]":
        return self.appended(other)
    
    def appended(self, other: "Network[Sy,Sz]") -> "Network[Sx,Sz]":
        if self.output_shape != other.input_shape:
            raise ValueError(f"Incompatible operand shape: {self.output_shape} != {other.input_shape}")
        return Network(
            device=self.device,
            input_shape=self.input_shape,
            output_shape=other.output_shape,
            logits=self.modules + other.modules
        )

    def save(self, path: str) -> None:
        return torch.save(self, path)

    @classmethod
    def load(cls, 
             path: str, 
             *, 
             device:        Device|None = None,
             input_shape:   Sx|None = None, 
             output_shape:  Sy|None = None) -> Self:
        with open(path, "rb") as file:
            if device is None:
                network: Network = torch.load(file)
            else:
                network = torch.load(file, map_location=get_device(device))

        if isinstance(network, cls):
            if input_shape and input_shape != network.input_shape:
                raise TypeError(f"Unpickled object has incorrect input-shape: {input_shape}, should be: {network.input_shape}")  
            
            if output_shape and output_shape != network.output_shape:
                raise TypeError(f"Unpickled object has incorrect output-shape: {output_shape}, should be: {network.output_shape}")  

            return cast(Self, network)
        else:
            raise TypeError(f"Unpickled object is of incorrect type: {type(network)}")