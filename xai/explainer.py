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

C = TypeVar("C", bound=Tuple[int,...])
F = TypeVar("F", bound=Tuple[int,...])

Explainers: TypeAlias = Literal[
    "exact",
    "permutation",
    "deep",
    "kernel"
]

class Explainer(Generic[C,F], ABC):

    @abstractmethod
    def __init__(self, network: "Network[F,C]", background: "Array", array_type: Type["Array"]) -> None:
        self._class_shape: C = network.output_shape
        self._class_size = math.prod(self._class_shape)
        
        self._feature_shape: F = network.input_shape
        self._feature_size = math.prod(self._feature_shape)
        
        self._array_type = array_type

        from . import Network
        
        input = Network.new(
            device=network.device,
            input_shape=(self._feature_size,),
        ).reshape(network.input_shape)

        output = Network.new(
            device=network.device,
            input_shape=network.output_shape
        ).reshape((self._class_size,))

        self._flattened_network = input + network + output
        self._background = self._feature_rows(background)

    def _to_numpy(self, array: "Array") -> NDArray[float64]:
        if isinstance(array, Tensor):
            array = cast(NDArray[Any], array.numpy(force=True))

        return array.astype(np.float64)
    
    def _to_tensor(self, array: "Array") -> Tensor:
        if isinstance(array, np.ndarray):
            array = cast(Tensor, torch.from_numpy(array))

        return array.to(dtype=torch.float32, device=self._flattened_network.device)
    
    def _feature_rows(self, array: "Array") -> "Array":
        if issubclass(self._array_type, np.ndarray):
            return self._to_numpy(array.reshape((-1,self._feature_size)))
        elif issubclass(self._array_type, Tensor):
            return self._to_tensor(array.reshape((-1,self._feature_size)))
        else:
            raise TypeError(f"Incompatible return type: {self._array_type}")
    
    def _call_network_func(self, array: "Array") -> NDArray[float64]:
        with torch.no_grad():
            return cast(np.ndarray, self._flattened_network(array).output().numpy()).astype(float64)
    
    @abstractmethod
    def _explain_single(self, flat_sample: "Array") -> Explanation[C,F]:
        pass

    def explain(self, samples: "Array", verbose: bool = False) -> Stream[Explanation[C,F]]:
        samples = self._feature_rows(samples)
        def iterator() -> Iterator[Explanation[C,F]]:
            with tqdm(total=len(samples), disable=not verbose) as bar:
                for i in range(len(samples)):
                    yield self._explain_single(samples[i:i+1])
                    bar.update()

        return Stream(iterator())

    @staticmethod
    def new(type: Explainers, network: "Network[F,C]", background: "Array") -> "Explainer[C,F]":
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

class ExactExplainer(Explainer[C,F]):

    def __init__(self, network: "Network[F,C]", background: "Array") -> None: 
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

    def _explain_single(self, flat_sample: "Array") -> Explanation[C,F]:
        explanation: shap.Explanation = self._explainer(flat_sample)[0]
        shap_values = np.moveaxis(explanation.values,0,1).reshape(self._class_shape + self._feature_shape)
        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=Seconds(explanation.compute_time),
            base_values=explanation.base_values,
            shap_values=shap_values,
        )

class PermutationExplainer(Explainer[C,F]):

    def __init__(self, network: "Network[F,C]", background: "Array") -> None:   
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

    def _explain_single(self, flat_sample: "Array") -> Explanation[C,F]:
        explanation: shap.Explanation = self._explainer(flat_sample)[0]
        shap_values = np.moveaxis(explanation.values,0,1).reshape(self._class_shape + self._feature_shape)
        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=Seconds(explanation.compute_time),
            base_values=explanation.base_values,
            shap_values=shap_values,
        )
        
class KernelExplainer(Explainer[C,F]):

    def __init__(self, network: "Network[F,C]", background: "Array") -> None:   
        super().__init__(
            network=network, 
            background=background, 
            array_type=np.ndarray
            )

        self._explainer = shap.KernelExplainer(
            model=self._call_network_func, 
            data=shap.kmeans(self._background, 100)
            )

    def _explain_single(self, flat_sample: "Array") -> Explanation[C,F]:
        explanation: shap.Explanation = self._explainer(flat_sample)[0]
        shap_values = np.moveaxis(explanation.values,0,1).reshape(self._class_shape + self._feature_shape)
        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=Seconds(explanation.compute_time),
            base_values=explanation.base_values,
            shap_values=shap_values,
        )

class DeepExplainer(Explainer[C,F]):

    def __init__(self, network: "Network[F,C]", background: "Array") -> None:    
        super().__init__(
            network=network, 
            background=background, 
            array_type=Tensor
            )

        self._explainer = shap.DeepExplainer(
            model=torch.nn.Sequential(*self._flattened_network.modules), 
            data=shap.sample(self._background)
            )
        self._base_values = torch.mean(self._flattened_network(self._background).output(), dim=0).numpy(force=True)

    def _explain_single(self, flat_sample: "Array") -> Explanation[C,F]:
        zero_time = Seconds.now()
        explanation = np.stack(self._explainer.shap_values(flat_sample), dtype=np.float64)[:,0,:]
        end_time = Seconds.now() - zero_time
        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=end_time,
            base_values=self._base_values,
            shap_values=explanation.reshape(self._class_shape + self._feature_shape),
        )