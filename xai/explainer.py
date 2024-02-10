from typing import *
from torch import Tensor
from numpy import float32, ndarray
from numpy.typing import NDArray

from .feed_forward import FeedForward
from .policy import Policy

import torch

class Explainer:
    
    def __init__(self, network: Policy, feed_forward: FeedForward) -> None:
        self._network = network
        self._feed_forward = feed_forward

    @overload
    def monopolar_saliency(self, 
                          with_respect_to: Callable[[Tensor],Tensor],
                          to_numpy: Literal[False],
                          order = 1) -> Tensor: ...

    @overload
    def monopolar_saliency(self, 
                          with_respect_to: Callable[[Tensor],Tensor],
                          to_numpy: Literal[True],
                          order = 1) -> NDArray[float32]: ...

    def monopolar_saliency(self, 
                          with_respect_to: Callable[[Tensor],Tensor],
                          to_numpy: bool,
                          order = 1) -> Tensor|NDArray[float32]:
        
        saliency = self._feed_forward.saliency(
            to_scalars=with_respect_to,
            order=order
            )
        
        if to_numpy:
            numpy_saliency = saliency.detach().cpu().numpy()
            assert isinstance(numpy_saliency, ndarray)
            return numpy_saliency
        else:
            return saliency
        
    @overload
    def bipolar_saliency(self, 
                         with_respect_to: Callable[[Tensor],Tensor],
                         to_numpy: Literal[False],
                         order = 1) -> Tensor:
        pass

    @overload
    def bipolar_saliency(self, 
                         with_respect_to: Callable[[Tensor],Tensor],
                         to_numpy: Literal[True],
                         order = 1) -> NDArray[float32]:
        pass
        
    def bipolar_saliency(self, 
                         with_respect_to: Callable[[Tensor],Tensor],
                         to_numpy: bool,
                         order = 1) -> Tensor|NDArray[float32]:
        derivative = self._feed_forward.derivative(
            to_scalars=with_respect_to,
            order=order
        )

        min,max = derivative.min(), derivative.max()
        saliency = torch.where(derivative < 0, derivative/min, derivative/max)

        if to_numpy:
            numpy_saliency = saliency.detach().cpu().numpy()
            assert isinstance(numpy_saliency, ndarray)
            return numpy_saliency
        else:
            return saliency