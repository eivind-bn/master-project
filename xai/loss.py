from typing import *
from torch import Tensor
from torch.nn import Module
from .stream import Stream

import torch
import warnings
import copy
import inspect

LossName: TypeAlias = Literal[
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    #"CTCLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    #"GaussianNLLLoss",
    "KLDivLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    #"MarginRankingLoss",
    "HingeEmbeddingLoss",
    #"MultiLabelMarginLoss",
    "HuberLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "MultiLabelSoftMarginLoss",
    #"CosineEmbeddingLoss",
    #"MultiMarginLoss",
    #"TripletMarginLoss",
    #"TripletMarginWithDistanceLoss",
]
LossSelector: TypeAlias = Callable[[Type["LossModule"]],"LossModule"]
Loss: TypeAlias = LossName|LossSelector|"LossModule"

T = TypeVar("T")
P = ParamSpec("P")

class LossType(Generic[P]):

    def __init__(self, loss_module: Callable[P,Module]) -> None:
        super().__init__()
        self.loss_module = loss_module

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "LossModule":
        return LossModule(self.loss_module(*args, **kwargs))
    

class LossModule(torch.nn.Module):
    L1Loss = LossType(torch.nn.L1Loss)
    MSELoss = LossType(torch.nn.MSELoss)
    CrossEntropyLoss = LossType(torch.nn.CrossEntropyLoss)
    #CTCLoss = LossType(torch.nn.CTCLoss)
    NLLLoss = LossType(torch.nn.NLLLoss)
    PoissonNLLLoss = LossType(torch.nn.PoissonNLLLoss)
    #GaussianNLLLoss = LossType(torch.nn.GaussianNLLLoss)
    KLDivLoss = LossType(torch.nn.KLDivLoss)
    BCELoss = LossType(torch.nn.BCELoss)
    BCEWithLogitsLoss = LossType(torch.nn.BCEWithLogitsLoss)
    #MarginRankingLoss = LossType(torch.nn.MarginRankingLoss)
    HingeEmbeddingLoss = LossType(torch.nn.HingeEmbeddingLoss)
    #MultiLabelMarginLoss = LossType(torch.nn.MultiLabelMarginLoss)
    HuberLoss = LossType(torch.nn.HuberLoss)
    SmoothL1Loss = LossType(torch.nn.SmoothL1Loss)
    SoftMarginLoss = LossType(torch.nn.SoftMarginLoss)
    MultiLabelSoftMarginLoss = LossType(torch.nn.MultiLabelSoftMarginLoss)
    #CosineEmbeddingLoss = LossType(torch.nn.CosineEmbeddingLoss)
    #MultiMarginLoss = LossType(torch.nn.MultiMarginLoss)
    #TripletMarginLoss = LossType(torch.nn.TripletMarginLoss)
    #TripletMarginWithDistanceLoss = LossType(torch.nn.TripletMarginWithDistanceLoss)

    def __init__(self, loss_module: Callable[[Tensor,Tensor],Tensor]) -> None:
        super().__init__()
        self._loss_module = loss_module

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self._loss_module(input, target)

    @classmethod
    def types(cls) -> Stream[Tuple[str,"LossType"]]:
        return Stream((name,var) for name,var in vars(cls).items() if isinstance(var, LossType))

    @overload
    @classmethod
    def get(cls, loss: Loss) -> "LossModule": ...

    @overload
    @classmethod
    def get(cls, loss: Loss, default: T) -> "LossModule"|T: ...

    @classmethod
    def get(cls, loss: Loss, default: T|None = None) -> "LossModule"|T:
        if isinstance(loss, str):
            for name,module_type in cls.types():
                if name == loss:
                    return module_type()
            
            if default is not None:
                return default
            else:
                raise KeyError(f"Loss-type: {loss} not found.")
        elif isinstance(loss, LossModule):
            return loss
        elif callable(loss):
            args = inspect.getfullargspec(loss).args
            match len(args):
                case 1:
                    return cast(LossSelector, loss)(LossModule)
                case _:
                    raise TypeError(f"Callable must accept 1 argument, but accepts {len(args)}")
        else:
            raise TypeError(f"Incompatible loss criterion: {type(loss)}")


# Verify that each loss function accepts two arguments once instantiated.
for name,module_type in LossModule.types():
    test_data_tensor = torch.ones((5,), dtype=torch.float32, device="cpu")
    try:
        test_target_tensor = torch.ones_like(test_data_tensor, dtype=torch.float32)
        with warnings.catch_warnings(action="ignore"):
            module = module_type() 
            loss = module(test_data_tensor,test_target_tensor)
            continue
    except Exception as e:
        pass
    try:
        test_target_tensor = torch.ones_like(test_data_tensor, dtype=torch.long)
        with warnings.catch_warnings(action="ignore"):
            module = module_type() 
            loss = module(test_data_tensor,torch.ones_like(test_data_tensor, dtype=torch.long))
            continue
    except Exception as e:
        raise ValueError(f"Incompatible module: {name}, desc: {e}")
