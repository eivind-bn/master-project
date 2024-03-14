from . import *
from torch import Tensor
from torch.nn import Module

import torch
import warnings
import inspect
import copy

ActivationName: TypeAlias = Literal[
    "ELU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardtanh",
    "Hardswish",
    "LeakyReLU",
    "LogSigmoid",
    #"MultiheadAttention",
    "PReLU",
    "ReLU",
    "ReLU6",
    "RReLU",
    "SELU",
    "CELU",
    "GELU",
    "Sigmoid",
    "SiLU",
    "Mish",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    #"Threshold",
    #"GLU",
    "Softmin",
    "Softmax",
    #"Softmax2d",
    "LogSoftmax",
    #"AdaptiveLogSoftmaxWithLoss",
]
ActivationSelector: TypeAlias = Callable[[Type["ActivationModule"]],"ActivationModule"]
Activation: TypeAlias = ActivationName|ActivationSelector|"ActivationModule"

T = TypeVar("T")
P = ParamSpec("P")

class ActivationType(Generic[P]):

    def __init__(self, activation_module: Callable[P,Module]) -> None:
        super().__init__()
        self.activation_module = activation_module

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "ActivationModule":
        return ActivationModule(self.activation_module(*args, **kwargs))


class ActivationModule(Module):
    ELU = ActivationType(torch.nn.ELU)
    Hardshrink = ActivationType(torch.nn.Hardshrink)
    Hardsigmoid = ActivationType(torch.nn.Hardsigmoid)
    Hardtanh = ActivationType(torch.nn.Hardtanh)
    Hardswish = ActivationType(torch.nn.Hardswish)
    LeakyReLU = ActivationType(torch.nn.LeakyReLU)
    LogSigmoid = ActivationType(torch.nn.LogSigmoid)
    #MultiheadAttention = ActivationType(torch.nn.MultiheadAttention)
    PReLU = ActivationType(torch.nn.PReLU)
    ReLU = ActivationType(torch.nn.ReLU)
    ReLU6 = ActivationType(torch.nn.ReLU6)
    RReLU = ActivationType(torch.nn.RReLU)
    SELU = ActivationType(torch.nn.SELU)
    CELU = ActivationType(torch.nn.CELU)
    GELU = ActivationType(torch.nn.GELU)
    Sigmoid = ActivationType(torch.nn.Sigmoid)
    SiLU = ActivationType(torch.nn.SiLU)
    Mish = ActivationType(torch.nn.Mish)
    Softplus = ActivationType(torch.nn.Softplus)
    Softshrink = ActivationType(torch.nn.Softshrink)
    Softsign = ActivationType(torch.nn.Softsign)
    Tanh = ActivationType(torch.nn.Tanh)
    Tanhshrink = ActivationType(torch.nn.Tanhshrink)
    #Threshold = ActivationType(torch.nn.Threshold)
    #GLU = ActivationType(torch.nn.GLU)
    Softmin = ActivationType(torch.nn.Softmin)
    Softmax = ActivationType(torch.nn.Softmax)
    #Softmax2d = ActivationType(torch.nn.Softmax2d)
    LogSoftmax = ActivationType(torch.nn.LogSoftmax)
    #AdaptiveLogSoftmaxWithLoss = ActivationType(torch.nn.AdaptiveLogSoftmaxWithLoss)


    def __init__(self, activation_module: Callable[[Tensor],Tensor]) -> None:
        super().__init__()
        self._activation_module = activation_module

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def __call__(self, input: Tensor) -> Tensor:
        return self._activation_module(input)
    
    def __repr__(self) -> str:
        if isinstance(self._activation_module, Module):
            return repr(self._activation_module)
        else:
            return f"{self.__class__.__name__}()"

    @classmethod
    def types(cls) -> Iterator[Tuple[str,"ActivationType"]]:
        for name,var in vars(cls).items():
            if isinstance(var, ActivationType):
                yield name, var

    @overload
    @classmethod
    def get(cls, activation: Activation) -> "ActivationModule": ...

    @overload
    @classmethod
    def get(cls, activation: Activation, default: T) -> "ActivationModule"|T: ...

    @classmethod
    def get(cls, activation: Activation, default: T|None = None) -> "ActivationModule"|T:
        if isinstance(activation, str):      
            for name,module_type in cls.types():
                if name == activation:
                    return module_type()
                
            if default is not None:
                return default
            else:
                raise KeyError(f"Activation-type: {activation} not found.")
        elif isinstance(activation, ActivationModule):
            return activation
        elif callable(activation):
            args = inspect.getfullargspec(activation).args
            match len(args):
                case 1:
                    return cast(ActivationSelector, activation)(ActivationModule)
                case _:
                    raise TypeError(f"Callable must accept 1 argument, but accepts {len(args)}")
        else:
            raise TypeError(f"Incompatible activation: {type(activation)}")

# Verify that each activation function accepts one arguments once instantiated.
for name,module_type in ActivationModule.types():
    try:
        test_tensor = torch.ones((5,), dtype=torch.float32, device="cpu")
        with warnings.catch_warnings(action="ignore"):
            module = module_type()
            activation = module(test_tensor)
    except Exception as e:
        raise ValueError(f"Incompatible module: {name}")