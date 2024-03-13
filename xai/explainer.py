from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, float64
from tqdm import tqdm
import math

from . import *

import shap # type: ignore
import numpy as np
import torch

if TYPE_CHECKING:
    from .network import Network

Array: TypeAlias = NDArray[Any]|Tensor
Sx = TypeVar("Sx", bound=Tuple[int,...])
Sy = TypeVar("Sy", bound=Tuple[int,...])

Explainers: TypeAlias = Literal[
    "permutation",
    "deep",
    "kernel"
]

class Explainer(Generic[Sx,Sy], ABC):

    @abstractmethod
    def __init__(self, network: "Network[Sx,Sy]", background: Array) -> None: ...

    @abstractmethod
    def explain(self, samples: Array, verbose: bool = False) -> Tuple[Explanation[Sx,Sy],...]:
        pass

    def inverse(self) -> Self[Sy,Sx]:
        pass

    @staticmethod
    def get_type(type: Explainers) -> Type[Explainer]:
        match type:
            case "permutation":
                return PermutationExplainer
            case "deep":
                return DeepExplainer
            case "kernel":
                return KernelExplainer
            case _:
                assert_never(type)

class PermutationExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: Array) -> None:    
        self.feature_shape = network.input_shape
        self.class_shape = network.output_shape
        self._network = network

        input_size = math.prod(self.feature_shape)
        if input_size > 128:
            raise ValueError(f"Network contain too many features: {input_size}")

        def forward(sample: Array) -> NDArray[float32]:
            numpy: NDArray[Any] = self._network(sample).output().numpy(force=True)
            return numpy.astype(float32)

        self._explainer = shap.PermutationExplainer(
            model=forward, 
            masker=self._to_numpy(background)
            )

    def _to_numpy(self, array: Array) -> NDArray[float32]:
        if isinstance(array, Tensor):
            array = cast(NDArray[Any], array.numpy(force=True))
        else:
            array = array
        
        return array.astype(float32).reshape((-1,) + self.feature_shape)

    def explain(self, samples: Array, verbose: bool = False) -> Tuple[Explanation[Sx,Sy],...]:
        explanations: List[Explanation[Sx,Sy]] = []
        samples = self._to_numpy(samples)
        with tqdm(total=len(samples), disable=not verbose) as bar:
            for i in range(len(samples)):
                explanations.append(Explanation(
                    feature_shape=self.feature_shape, 
                    class_shape=self.class_shape, 
                    explanation=self._explainer(samples[i:i+1]))
                    )
                bar.update()

        return tuple(explanations)


class ExactExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: Array) -> None:    
        self.feature_shape = network.input_shape
        self.class_shape = network.output_shape
        self._flattened_network = network.flatten()

        input_size = math.prod(self.feature_shape)
        if input_size > 16:
            raise ValueError(f"Network contain too many features: {input_size}")

        def forward(sample: Array) -> NDArray[float32]:
            numpy: NDArray[Any] = self._flattened_network(sample).output().numpy(force=True)
            return numpy.astype(float32)

        self._explainer = shap.ExactExplainer(
            model=forward, 
            masker=self._to_numpy(background)
            )

    def _to_numpy(self, array: Array) -> NDArray[float32]:
        if isinstance(array, Tensor):
            array = cast(NDArray[Any], array.numpy(force=True))
        else:
            array = array
        
        return array.astype(float32).reshape((-1,) + self.feature_shape)

    def explain(self, samples: Array, verbose: bool = False) -> Tuple[Explanation[Sx,Sy],...]:
        explanations: List[Explanation[Sx,Sy]] = []
        samples = self._to_numpy(samples)
        with tqdm(total=len(samples), disable=not verbose) as bar:
            for i in range(len(samples)):
                explanations.append(Explanation(
                    feature_shape=self.feature_shape, 
                    class_shape=self.class_shape, 
                    explanation=self._explainer(samples[i:i+1])
                    ))
                bar.update()

        return tuple(explanations)


class KernelExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: Array) -> None:    
        from .network import Network
        self.feature_shape = network.input_shape
        self.class_shape = network.output_shape
        self.input_size = math.prod(self.feature_shape)
        self._flattened_network = Network.new(
            device=network.device,
            input_shape=(self.input_size,),
        ).reshape(self.feature_shape) + network.flatten()

        def forward(sample: Array) -> NDArray[float32]:
            numpy: NDArray[Any] = self._flattened_network(sample).output().numpy(force=True)
            return numpy.astype(float32)

        self._explainer = shap.KernelExplainer(
            model=forward, 
            data=shap.kmeans(self._to_numpy(background), 100)
            )

    def _to_numpy(self, array: Array) -> NDArray[float32]:
        if isinstance(array, Tensor):
            array = cast(NDArray[Any], array.numpy(force=True))
        else:
            array = array
        
        return array.astype(float32).reshape((-1,self.input_size))

    def explain(self, samples: Array, verbose: bool = False) -> Tuple[Explanation[Sx,Sy],...]:
        explanations: List[Explanation[Sx,Sy]] = []
        samples = self._to_numpy(samples)
        with tqdm(total=len(samples), disable=not verbose) as bar:
            for i in range(len(samples)):
                explanations.append(Explanation(
                    feature_shape=self.feature_shape, 
                    class_shape=self.class_shape, 
                    explanation=self._explainer(samples[i:i+1])
                    ))
                bar.update()

        return tuple(explanations)

class DeepExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: Array) -> None:    
        self.feature_shape = network.input_shape
        self.class_shape = network.output_shape
        self.input_size = math.prod(self.feature_shape)
        self._network = network

        self._explainer = shap.DeepExplainer(
            model=torch.nn.Sequential(*network.modules()), 
            data=shap.sample(self._to_tensor(background), 100)
            )

    def _to_tensor(self, array: Array) -> torch.Tensor:
        if isinstance(array, Tensor):
            array = cast(NDArray[Any], array.numpy(force=True))
        else:
            array = array

        array = array.astype(float32).reshape((-1,self.input_size))
        
        return torch.from_numpy(array).to(device=self._network.device, dtype=torch.float32)

    def explain(self, samples: Array, verbose: bool = False) -> Tuple[Explanation[Sx,Sy],...]:
        explanations: List[Explanation[Sx,Sy]] = []
        samples = self._to_tensor(samples)
        with tqdm(total=len(samples), disable=not verbose) as bar:
            for i in range(len(samples)):
                explanation = np.stack(self._explainer.shap_values(samples[i:i+1]))
                explanations.append(explanation)
                bar.update()

        return tuple(explanations)