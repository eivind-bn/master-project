from typing import *
from abc import ABC, abstractmethod
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, float64
from tqdm import tqdm
import math
from .time import Seconds

import shap # type: ignore

if TYPE_CHECKING:
    from .network import Network

Array: TypeAlias = NDArray[Any]|Tensor
Ints: TypeAlias = Tuple[int,...]
Sx = TypeVar("Sx", bound=Ints)
Sy = TypeVar("Sy", bound=Ints)

class Explanation(Generic[Sx,Sy]):

    def __init__(self, 
                 input_shape:   Sx, 
                 output_shape:  Sy, 
                 explanation:   shap.Explanation) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.compute_time = Seconds(explanation.compute_time)
        self.values: NDArray[float64] = cast(NDArray[float64], explanation.values).reshape(self.input_shape + self.output_shape)
        

class Explainer(Generic[Sx,Sy], ABC):

    @abstractmethod
    def __init__(self, network: "Network[Sx,Sy]", background: Array) -> None: ...

    @abstractmethod
    def explain(self, samples: Array, verbose: bool = False) -> Tuple[Explanation[Sx,Sy],...]:
        pass

class PermutationExplainer(Explainer[Sx,Sy]):

    def __init__(self, network: "Network[Sx,Sy]", background: Array) -> None:    
        self.input_shape = network.input_shape
        self.output_shape = network.output_shape
        self._flattened_network = network.flatten()

        input_size = math.prod(self.input_shape)
        if input_size > 128:
            raise ValueError(f"Network contain too many features: {input_size}")

        def forward(sample: Array) -> NDArray[float32]:
            numpy: NDArray[Any] = self._flattened_network(sample).output().numpy(force=True)
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
        
        return array.astype(float32).reshape((-1,) + self.input_shape)

    def explain(self, samples: Array, verbose: bool = False) -> Tuple[Explanation[Sx,Sy],...]:
        explanations: List[Explanation[Sx,Sy]] = []
        samples = self._to_numpy(samples)
        with tqdm(total=len(samples), disable=not verbose) as bar:
            for i in range(len(samples)):
                explanations.append(Explanation(self.input_shape, self.output_shape, self._explainer(samples[i:i+1])))
                bar.update()

        return tuple(explanations)



