from . import *
from abc import ABC, abstractmethod
from torch import Tensor
from numpy.typing import NDArray
from numpy import float64
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
    "kernel",
    "gradient"
]

class Explainer(Generic[C,F], ABC):

    def __init__(self, 
                 network:       "Network[F,C]", 
                 background:    "Array") -> None:
        
        self._class_shape: C = network.output_shape
        self._class_size = math.prod(self._class_shape)
        
        self._feature_shape: F = network.input_shape
        self._feature_size = math.prod(self._feature_shape)

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
        self._explainer = self._init_explainer(background)


    @property
    @abstractmethod
    def _array_type(self) -> "Type[Array]":
        pass

    @abstractmethod
    def _init_explainer(self, background: "Array") -> shap.Explainer:
        pass

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
            return cast(np.ndarray, self._flattened_network(array).output().cpu().numpy()).astype(float64)
    
    @abstractmethod
    def _explain_single(self, 
                        flat_sample:    "Array",
                        max_evals:      int|None = None) -> Explanation[C,F]:
        pass

    def explain(self, 
                samples:    "Array", 
                max_evals:  int|None = None, 
                verbose:    bool = False) -> Stream[Explanation[C,F]]:
        samples = self._feature_rows(samples)
        def iterator() -> Iterator[Explanation[C,F]]:
            with tqdm(total=len(samples), disable=not verbose) as bar:
                for i in range(len(samples)):
                    yield self._explain_single(samples[i:i+1], max_evals)
                    bar.update()

        return Stream(iterator())

class ExactExplainer(Explainer[C,F]):

    def __init__(self, 
                 network:       "Network[F,C]", 
                 background:    "Array") -> None:
        super().__init__(network, background)

    @property
    def _array_type(self) -> Type[np.ndarray]:
        return np.ndarray
        
    def _init_explainer(self, background: "Array") -> shap.Explainer:
        return shap.ExactExplainer(
            model=self._call_network_func,
            masker=self._feature_rows(background)
            )

    def _explain_single(self, 
                        flat_sample:    "Array", 
                        max_evals:      int|None = 100_000) -> Explanation[C,F]:
                        
        if max_evals is None:
            max_evals = 2**self._feature_size

        self._explainer: shap.ExactExplainer
        explanation: shap.Explanation = self._explainer(
            flat_sample,
            max_evals=max_evals
        )

        shap_values = np.reshape(explanation.values, (self._feature_size, self._class_size))
        shap_values = np.moveaxis(shap_values, 0, 1)
        shap_values = shap_values.reshape(self._class_shape + self._feature_shape)

        base_values = np.reshape(explanation.base_values, (self._class_size,))
        base_values = base_values.reshape(self._class_shape)

        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=Seconds(explanation.compute_time),
            shap_values=shap_values,
        )

class PermutationExplainer(Explainer[C,F]):

    def __init__(self, 
                 network:       "Network[F,C]", 
                 background:    "Array") -> None:
        super().__init__(network, background)

    @property
    def _array_type(self) -> Type[np.ndarray]:
        return np.ndarray

    def _init_explainer(self, background: "Array") -> shap.Explainer:
        return shap.PermutationExplainer(
            model=self._call_network_func, 
            masker=self._feature_rows(background),
            )

    def _explain_single(self, 
                        flat_sample:    "Array", 
                        max_evals:      int|None = 100_000) -> Explanation[C,F]:
        
        if max_evals is None:
            max_evals = 2*self._feature_size + 1

        self._explainer: shap.PermutationExplainer
        explanation: shap.Explanation = self._explainer(
            flat_sample, 
            max_evals=max_evals
            )

        shap_values = np.reshape(explanation.values, (self._feature_size, self._class_size))
        shap_values = np.moveaxis(shap_values, 0, 1)
        shap_values = shap_values.reshape(self._class_shape + self._feature_shape)

        base_values = np.reshape(explanation.base_values, (self._class_size,))
        base_values = base_values.reshape(self._class_shape)

        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=Seconds(explanation.compute_time),
            shap_values=shap_values,
        )
        
class KernelExplainer(Explainer[C,F]):

    def __init__(self, 
                 network:       "Network[F,C]", 
                 background:    "Array") -> None:
        super().__init__(network, background)
        
    @property
    def _array_type(self) -> Type[np.ndarray]:
        return np.ndarray
        
    def _init_explainer(self, background: "Array") -> shap.Explainer:
        return shap.KernelExplainer(
            model=self._call_network_func, 
            data=shap.kmeans(self._feature_rows(background), 100),
            )

    def _explain_single(self, 
                        flat_sample:    "Array", 
                        max_evals:      int|None = 100_000) -> Explanation[C,F]:
        
        
        if max_evals is None:
            max_evals = 2*self._feature_size + 2048

        self._explainer: shap.KernelExplainer

        zero_time = Seconds.now()
        shap_values: shap.Explanation = self._explainer.shap_values(flat_sample, nsamples=max_evals)
        end_time = Seconds.now() - zero_time

        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)

        if hasattr(self._explainer.expected_value, "__len__"):
            base_values = np.tile(self._explainer.expected_value, (shap_values.shape[0],1))
        else:
            base_values = np.tile(self._explainer.expected_value, shap_values.shape[0])
        
        shap_values = np.reshape(shap_values, (self._feature_size, self._class_size))
        shap_values = np.moveaxis(shap_values, 0, 1)
        shap_values = shap_values.reshape(self._class_shape + self._feature_shape)

        base_values = np.reshape(self._explainer.expected_value, (self._class_size,))
        base_values = base_values.reshape(self._class_shape)

        
        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=end_time,
            shap_values=shap_values,
        )

class DeepExplainer(Explainer[C,F]):

    @property
    def _array_type(self) -> Type[Tensor]:
        return Tensor
    
    @property
    def base_values(self) -> NDArray[float64]:
        with torch.no_grad():
            if not hasattr(self, "_base_values"):
                expected_value: np.ndarray = self._explainer.explainer.expected_value
                self._base_values = expected_value.astype(np.float64)

            return self._base_values

    def _init_explainer(self, background: "Array") -> shap.Explainer:
        return shap.DeepExplainer(
            model=torch.nn.Sequential(*self._flattened_network.modules), 
            data=shap.sample(self._feature_rows(background))
            )

    def _explain_single(self, 
                        flat_sample:    "Array", 
                        max_evals:      int|None = None) -> Explanation[C,F]:

        zero_time = Seconds.now()

        self._explainer: shap.DeepExplainer
        explanation = np.stack(self._explainer.shap_values(flat_sample), dtype=np.float64)
        explanation = explanation.reshape(self._class_shape + self._feature_shape)
        end_time = Seconds.now() - zero_time
        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=end_time,
            shap_values=explanation,
        )
    
class GradientExplainer(Explainer[C,F]):

    @property
    def _array_type(self) -> Type[Tensor]:
        return Tensor
    
    @property
    def base_values(self) -> NDArray[float64]:
        with torch.no_grad():
            if not hasattr(self, "_base_values"):
                expected_value: np.ndarray = np.mean(self._flattened_network(self._feature_rows(self._background)).output().numpy(force=True), axis=0)
                self._base_values = expected_value.astype(np.float64)

            return self._base_values


    def _init_explainer(self, background: "Array") -> shap.Explainer:
        return shap.GradientExplainer(
            model=torch.nn.Sequential(*self._flattened_network.modules), 
            data=shap.sample(self._feature_rows(background))
            )

    def _explain_single(self, 
                        flat_sample:    "Array", 
                        max_evals:      int|None = 2**16) -> Explanation[C,F]:
        
        if max_evals is None:
            max_evals = 2**self._feature_size

        zero_time = Seconds.now()

        self._explainer: shap.GradientExplainer
        explanation = np.stack(self._explainer.shap_values(flat_sample), dtype=np.float64)
        explanation = explanation.reshape(self._class_shape + self._feature_shape)
        end_time = Seconds.now() - zero_time
        return Explanation(
            class_shape=self._class_shape,
            feature_shape=self._feature_shape,
            compute_time=end_time,
            shap_values=explanation,
        )