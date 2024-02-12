from typing import *
from torch import Tensor
from numpy import float32, ndarray
from numpy.typing import NDArray
from shap import PermutationExplainer, Explanation

from .feed_forward import FeedForward
from .policy import Policy

import torch

SaliencyType = Literal["Absolute", "Signed"]
ShapType = Literal["Permutation"]

class Explainer:
    
    def __init__(self, network: Policy, feed_forward: FeedForward) -> None:
        self._network = network
        self._feed_forward = feed_forward
        self._permutation_explainer: PermutationExplainer|None = None

    @overload
    def saliency(self, 
                 with_respect_to:   Callable[[Tensor],Tensor],
                 saliency_type:     SaliencyType,
                 to_numpy:          Literal[False],
                 order:             int = 1) -> Tensor: ...

    @overload
    def saliency(self, 
                 with_respect_to:   Callable[[Tensor],Tensor],
                 saliency_type:     SaliencyType,
                 to_numpy:          Literal[True],
                 order:             int = 1) -> NDArray[float32]: ...

    def saliency(self, 
                 with_respect_to:   Callable[[Tensor],Tensor],
                 saliency_type:     SaliencyType,
                 to_numpy:          bool,
                 order:             int = 1) -> Tensor|NDArray[float32]:
        match saliency_type:
            case "Absolute":  
                saliency = self._feed_forward.saliency(
                    to_scalars=with_respect_to,
                    order=order
                    )
            case "Signed":
                derivative = self._feed_forward.derivative(
                    to_scalars=with_respect_to,
                    order=order
                    )
                
                min,max = derivative.min(), derivative.max()
                saliency = torch.where(derivative < 0, derivative/min, derivative/max)
            case _:
                assert_never(saliency_type)

        if to_numpy:
            numpy_saliency = saliency.detach().cpu().numpy()
            assert isinstance(numpy_saliency, ndarray)
            return numpy_saliency
        else:
            return saliency
        
    def shap(self, shap_type: ShapType) -> NDArray[float32]:
        if self._permutation_explainer is None:
            self._permutation_explainer = PermutationExplainer(self._network.network, )