from . import *
from abc import ABC, abstractmethod
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, float64
from tqdm import tqdm

import math
import shap # type: ignore
import numpy as np
import torch

if TYPE_CHECKING:
    from .network import Network, Array

Sx = TypeVar("Sx", bound=Tuple[int,...])
Sy = TypeVar("Sy", bound=Tuple[int,...])

Explainers: TypeAlias = Literal[
    "exact",
    "permutation",
    "deep",
    "kernel"
]

class Explainer(Generic[Sx,Sy], ABC):

    @abstractmethod
    def __init__(self, network: "Network[Sx,Sy]", background: "Array", array_type: Type["Array"]) -> None:
        self._feature_shape = network.input_shape
        self._class_shape = network.output_shape
        self._network = network
        self._feature_size = math.prod(self._feature_shape)
        self._class_size = math.prod(self._class_shape)
        self._array_type = array_type
        self._background = self._feature_rows(background)

    def _to_numpy(self, array: "Array") -> NDArray[float64]:
        if isinstance(array, Tensor):
            array = cast(NDArray[Any], array.numpy(force=True))

        return array.astype(np.float64)
    
    def _to_tensor(self, array: "Array") -> Tensor:
        if isinstance(array, np.ndarray):
            array = cast(Tensor, torch.from_numpy(array))

        return array.to(dtype=torch.float32, device=self._network.device)
    
    def _feature_rows(self, array: "Array") -> "Array":
        if issubclass(self._array_type, np.ndarray):
            return self._to_numpy(array.reshape((-1,self._feature_size)))
        elif issubclass(self._array_type, Tensor):
            return self._to_tensor(array.reshape((-1,self._feature_size)))
        else:
            raise TypeError(f"Incompatible return type: {self._array_type}")
    
    def _call_network_func(self, array: "Array") -> "Array":
        array = self._to_tensor(array).reshape((-1,) + self._feature_shape)
        output = self._network(array).output().reshape((-1,self._class_size))
        if issubclass(self._array_type, np.ndarray):
            return self._to_numpy(output)
        elif issubclass(self._array_type, Tensor):
            return output
        else:
            raise TypeError(f"Incompatible return type: {self._array_type}")
    
    @abstractmethod
    def _explain_single(self, sample: "Array") -> Explanation[Sx,Sy]:
        pass

    def explain(self, samples: "Array", verbose: bool = False) -> Stream[Explanation[Sx,Sy]]:
        samples = self._feature_rows(samples)
        def iterator() -> Iterator[Explanation[Sx,Sy]]:
            with tqdm(total=len(samples), disable=not verbose) as bar:
                for i in range(len(samples)):
                    yield self._explain_single(samples[i:i+1])
                    bar.update()

        return Stream(iterator())

    @staticmethod
    def new(type: Explainers, network: "Network[Sx,Sy]", background: "Array") -> "Explainer[Sx,Sy]":
        match type:
            case "exact":
                return ExactExplainer(network, background)
            case "permutation":
                return PermutationExplainer(network, background)
            case "deep":
                return DeepExplainer(network, background)
            case "kernel":
                return KernelExplainer(network, background)
            case _:
                assert_never(type)

class ExactExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: "Array") -> None: 
        super().__init__(
            network=network, 
            background=background, 
            array_type=np.ndarray
            )

        self._explainer = shap.ExactExplainer(
            model=self._call_network_func,
            masker=self._background
            )
        
        if self._feature_size > 16:
            raise ValueError(f"Network contain too many features: {self._feature_size}")

    def _explain_single(self, sample: "Array") -> Explanation[Sx,Sy]:
        explanation: shap.Explanation = self._explainer(sample)
        return Explanation(
            feature_shape=self._feature_shape,
            class_shape=self._class_shape,
            explanation=explanation
        )

class PermutationExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: "Array") -> None:   
        super().__init__(
            network=network, 
            background=background, 
            array_type=np.ndarray
            )

        self._explainer = shap.PermutationExplainer(
            model=self._call_network_func, 
            masker=self._background
            )
        
        if self._feature_size > 128:
            raise ValueError(f"Network contain too many features: {self._feature_size}")

    def _explain_single(self, sample: "Array") -> Explanation[Sx,Sy]:
        explanation: shap.Explanation = self._explainer(sample)
        return Explanation(
            feature_shape=self._feature_shape,
            class_shape=self._class_shape,
            explanation=explanation
        )
        
class KernelExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: "Array") -> None:   
        super().__init__(
            network=network, 
            background=background, 
            array_type=np.ndarray
            )

        self._explainer = shap.KernelExplainer(
            model=self._call_network_func, 
            data=shap.kmeans(self._background, 100)
            )

    def _explain_single(self, sample: "Array") -> Explanation[Sx,Sy]:
        explanation: shap.Explanation = self._explainer(sample)
        return Explanation(
            feature_shape=self._feature_shape,
            class_shape=self._class_shape,
            explanation=explanation
        )

class DeepExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: "Array") -> None:    
        super().__init__(
            network=network, 
            background=background, 
            array_type=Tensor
            )
        from . import Network
        
        input = Network.new(
            device=network.device,
            input_shape=(self._feature_size,),
        ).reshape(network.input_shape)

        output = Network.new(
            device=network.device,
            input_shape=network.output_shape
        ).reshape((self._class_size,))

        flattened_network = input + network + output

        self._explainer = shap.DeepExplainer(
            model=torch.nn.Sequential(*flattened_network.modules()), 
            data=shap.sample(self._background)
            )
        self._base_values = torch.mean(network(background).output(), dim=0).numpy(force=True)

    def _explain_single(self, sample: "Array") -> Explanation[Sx,Sy]:
        zero_time = Seconds.now()
        explanation = np.stack(self._explainer.shap_values(sample), dtype=np.float64)
        end_time = Seconds.now() - zero_time
        return Explanation(
            feature_shape=self._feature_shape,
            class_shape=self._class_shape,
            compute_time=end_time,
            shap_values=explanation,
            base_values=self._base_values
        )