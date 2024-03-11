from __future__ import annotations
from typing import *
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, float64
import math
from .time import Seconds

import shap # type: ignore
import numpy as np

Array: TypeAlias = NDArray[Any]|Tensor
Ints: TypeAlias = Tuple[int,...]
Int: TypeAlias = Tuple[int]
Sx = TypeVar("Sx", bound=Ints)
Sy = TypeVar("Sy", bound=Ints)
Sa = TypeVar("Sa", bound=Ints)
Sb = TypeVar("Sb", bound=Ints)

class Explanation(Generic[Sx,Sy]):

    @overload
    def __init__(self,
                 *,
                 input_shape:   Sx, 
                 output_shape:  Sy, 
                 compute_time:  Seconds,
                 shap_values:   NDArray[float64]) -> None: ...
        
    @overload
    def __init__(self,
                 *,
                 input_shape:   Sx, 
                 output_shape:  Sy, 
                 explanation:   shap.Explanation) -> None: ...

    def __init__(self,
                 *,
                 input_shape:   Sx, 
                 output_shape:  Sy, 
                 compute_time:  Seconds|None = None,
                 shap_values:   NDArray[float64]|None = None,
                 explanation:   shap.Explanation|None = None) -> None:
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape

        if compute_time is None or shap_values is None:
            if explanation is None:
                raise ValueError()
            
            self._shap_values: NDArray[float64] = explanation.values
            self._compute_time = Seconds(explanation.compute_time)

        else:
            if explanation is not None:
                raise ValueError()
                      
            self._compute_time = compute_time
            self._shap_values = shap_values.reshape(self.input_shape + self.output_shape)

    @property
    def input_shape(self) -> Sx:
        return self._input_shape

    @property
    def output_shape(self) -> Sy:
        return self._output_shape

    @property
    def shap_values(self) -> NDArray[float64]:
        return self._shap_values

    @property
    def compute_time(self) -> Seconds:
        return self._compute_time

    @property
    def T(self) -> Explanation[Sy,Sx]:
        return self._compute(
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            shap_values=self.shap_values.T
        )
    
    def reshape(self, 
                input_shape:    Sa, 
                output_shape:   Sb) -> Explanation[Sa,Sb]:
        return self._compute(
            input_shape=input_shape,
            output_shape=output_shape,
            shap_values=self.shap_values.reshape(input_shape + output_shape)
        )
    
    def flatten(self) -> Explanation[Int,Int]:
        return self.reshape(
            input_shape=(math.prod(self.input_shape),),
            output_shape=(math.prod(self.output_shape),)
        )

    def combine(self, other: Explanation[Ints,Sb]) -> Explanation[Sy,Sb]:
        A = self.flatten()
        B = other.flatten()
        return (A.T @ B).reshape(
            input_shape=self.output_shape,
            output_shape=other.output_shape
        )
    
    def max(self) -> float:
        return float(self.shap_values.max())
    
    def abs(self) -> Explanation[Sx,Sy]:
        return Explanation(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            compute_time=self.compute_time,
            shap_values=np.abs(self.shap_values)
        )
    
    def to_rgb(self: Explanation[Sx,Tuple[int,int]], norm: float|None = None) -> NDArray[float32]:
        if norm is None:
            norm = self.abs().max()
        else:
            norm = abs(norm)

        norm = abs(norm)
        rgb = np.zeros(self.input_shape + self.output_shape + (3,), dtype=np.float32)
        black = np.zeros_like(self.shap_values)
        red = np.where(self.shap_values > 0, self.shap_values/norm, black)
        blue = np.where(self.shap_values < 0, -self.shap_values/norm, black)
        rgb[:,:,:,0] = red
        rgb[:,:,:,2] = blue
        return rgb
    
    def __add__(self, other: Explanation[Sx,Sy]) -> Explanation[Sx,Sy]:
        return self._compute(shap_values=self.shap_values + other.shap_values)
    
    def __sub__(self, other: Explanation[Sx,Sy]) -> Explanation[Sx,Sy]:
        return self._compute(shap_values=self.shap_values - other.shap_values)
    
    def __mul__(self, other: Explanation[Sx,Sy]) -> Explanation[Sx,Sy]:
        return self._compute(shap_values=self.shap_values * other.shap_values)
    
    def __truediv__(self, other: Explanation[Sx,Sy]) -> Explanation[Sx,Sy]:
        return self._compute(shap_values=self.shap_values / other.shap_values)

    def __matmul__(self, other: Explanation[Sy,Sa]) -> Explanation[Sx,Sa]:
        return self._compute(
            output_shape=other.output_shape,
            shap_values=self.shap_values @ other.shap_values
        )
    
    @overload
    def _compute(self,
                 *,
                 shap_values:   NDArray[float64] = ...) -> Explanation[Sx,Sy]: ...

    @overload
    def _compute(self,
                 *,
                 input_shape:   Sa, 
                 shap_values:   NDArray[float64] = ...) -> Explanation[Sa,Sy]: ...
    
    @overload
    def _compute(self,
                 *,
                 output_shape:  Sb,
                 shap_values:   NDArray[float64] = ...) -> Explanation[Sx,Sb]: ...

    @overload  
    def _compute(self,
                 *,
                 input_shape:   Sa, 
                 output_shape:  Sb,
                 shap_values:   NDArray[float64] = ...) -> Explanation[Sa,Sb]: ...
    
    
    def _compute(self,
                 *,
                 input_shape:   Sa|None = None, 
                 output_shape:  Sb|None = None,
                 shap_values:   NDArray[float64]|None = None) -> Explanation:
        return Explanation(
            input_shape=self.input_shape if input_shape is None else input_shape,
            output_shape=self.output_shape if output_shape is None else output_shape,
            compute_time=self.compute_time,
            shap_values=self.shap_values if shap_values is None else shap_values
        )
