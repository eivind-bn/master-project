from typing import *
from torch import Tensor
from torch.nn import Module

import torch
import warnings

LossName = Literal[
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    #"CTCLoss",
    #"NLLLoss",
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
LossSelector = Callable[[Type["LossModule"]],"LossModule"]
LossFunction = Callable[[Tensor,Tensor],Tensor]

T = TypeVar("T")
P = ParamSpec("P")

class LossType(Generic[P]):

    def __init__(self, loss_module: Callable[P,Module]) -> None:
        super().__init__()
        self.loss_module = loss_module

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "LossModule":
        return LossModule(self.loss_module(*args, **kwargs))
    

class LossModule(Generic[P]):
    L1Loss = LossType(torch.nn.L1Loss)
    MSELoss = LossType(torch.nn.MSELoss)
    CrossEntropyLoss = LossType(torch.nn.CrossEntropyLoss)
    #CTCLoss = LossType(torch.nn.CTCLoss)
    #NLLLoss = LossType(torch.nn.NLLLoss)
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

    def __init__(self, loss_module: Module|Callable[[Tensor,Tensor],Tensor]) -> None:
        super().__init__()
        self._loss_module = loss_module

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return cast(Tensor, self._loss_module(input, target))

    @classmethod
    def types(cls) -> Iterator[Tuple[str,"LossType"]]:
        for name,var in vars(cls).items():
            if isinstance(var, LossType):
                yield name, var

    @overload
    @classmethod
    def get(cls, loss_name: LossName, *, default: T) -> "LossType"|T: ...

    @overload
    @classmethod
    def get(cls, loss_name: LossName) -> "LossType": ...

    @classmethod
    def get(cls, loss_name: LossName, **kwargs: T) -> "LossType"|T:
        for name,module_type in cls.types():
            if name == loss_name:
                return module_type
        
        if "default" in kwargs:
            return kwargs["default"]
        else:
            raise KeyError(f"Loss-type: {loss_name} not found.")

# Verify that each loss function accepts two arguments once instantiated.
for name,module_type in LossModule.types():
    try:
        test_tensor = torch.ones((5,), dtype=torch.float32, device="cpu")
        with warnings.catch_warnings(action="ignore"):
            module = module_type() 
            loss = module(test_tensor,test_tensor)
    except Exception as e:
        raise ValueError(f"Incompatible module: {name}")
