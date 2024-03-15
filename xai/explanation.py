from . import *
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, float64

import shap # type: ignore
import numpy as np
import math

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
                 feature_shape: Sx, 
                 class_shape:   Sy, 
                 compute_time:  Seconds,
                 base_values:   NDArray[float64],
                 shap_values:   NDArray[float64]) -> None: ...
        
    @overload
    def __init__(self,
                 *,
                 feature_shape: Sx, 
                 class_shape:   Sy, 
                 explanation:   shap.Explanation) -> None: ...

    def __init__(self,
                 *,
                 feature_shape: Sx, 
                 class_shape:   Sy, 
                 compute_time:  Seconds|None = None,
                 base_values:   NDArray[float64]|None = None,
                 shap_values:   NDArray[float64]|None = None,
                 explanation:   shap.Explanation|None = None) -> None:
        super().__init__()
        self._feature_shape = feature_shape
        self._class_shape = class_shape

        if compute_time is None or shap_values is None or base_values is None:
            if explanation is None:
                raise ValueError()
            
            self._shap_values: NDArray[float64] = explanation.values
            self._base_values: NDArray[float64] = explanation.base_values
            self._compute_time = Seconds(explanation.compute_time)
            self._explanation = explanation

            assert self._base_values is not None

        else:
            if explanation is not None:
                raise ValueError()
                      
            self._shap_values = shap_values
            self._base_values = base_values
            self._compute_time = compute_time
            self._explanation = None

        self._shap_values = self._shap_values.reshape(self.feature_shape + self.class_shape)

    @property
    def feature_shape(self) -> Sx:
        return self._feature_shape

    @property
    def class_shape(self) -> Sy:
        return self._class_shape

    @property
    def shap_values(self) -> NDArray[float64]:
        return self._shap_values
    
    @property
    def base_values(self) -> NDArray[float64]:
        return self._base_values

    @property
    def compute_time(self) -> Seconds:
        return self._compute_time

    @property
    def T(self) -> "Explanation[Sy,Sx]":
        return self._compute(
            feature_shape=self.class_shape,
            class_shape=self.feature_shape,
            shap_values=self.shap_values.T
        )
    
    def reshape(self, 
                feature_shape:    Sa, 
                class_shape:   Sb) -> "Explanation[Sa,Sb]":
        return self._compute(
            feature_shape=feature_shape,
            class_shape=class_shape,
            shap_values=self.shap_values.reshape(feature_shape + class_shape)
        )
    
    def flatten(self) -> "Explanation[Int,Int]":
        return self.reshape(
            feature_shape=(math.prod(self.feature_shape),),
            class_shape=(math.prod(self.class_shape),)
        )

    def combine(self, other: "Explanation[Sx,Sb]") -> "Explanation[Sy,Sb]":
        A = self.flatten()
        B = other.flatten()
        return (A.T @ B).reshape(
            feature_shape=self.class_shape,
            class_shape=other.class_shape
        )
    
    def max(self) -> float:
        return float(self.shap_values.max())
    
    def abs(self) -> "Explanation[Sx,Sy]":
        return Explanation(
            feature_shape=self.feature_shape,
            class_shape=self.class_shape,
            compute_time=self.compute_time,
            shap_values=np.abs(self.shap_values),
            base_values=self.base_values
        )
    
    def feature_sum(self) -> "Explanation[Tuple[Literal[1]],Sy]":
        axes = tuple(range(len(self.feature_shape)))
        values = self.shap_values.sum(axes)
        return Explanation(
            feature_shape=(1,),
            class_shape=self.class_shape,
            compute_time=self.compute_time,
            shap_values=values,
            base_values=self.base_values
        )
    
    def class_sum(self) -> "Explanation[Sx,Tuple[Literal[1]]]":
        feature_axis_cnt = len(self.feature_shape)
        class_axis_cnt = len(self.class_shape)
        axes = tuple(range(feature_axis_cnt, feature_axis_cnt+class_axis_cnt))
        values = self.shap_values.sum(axes)
        return Explanation(
            feature_shape=self.feature_shape,
            class_shape=(1,),
            compute_time=self.compute_time,
            shap_values=values,
            base_values=self.base_values
        )
    
    def to_rgb(self: "Explanation[Sx,Tuple[int,int]]", norm: float|None = None) -> NDArray[float32]:
        if norm is None:
            norm = self.abs().max()
        else:
            norm = abs(norm)

        norm = abs(norm)
        rgb = np.zeros(self.feature_shape + self.class_shape + (3,), dtype=np.float32)
        black = np.zeros_like(self.shap_values)
        red = np.where(self.shap_values > 0, self.shap_values/norm, black)
        blue = np.where(self.shap_values < 0, -self.shap_values/norm, black)
        rgb[:,:,:,0] = red
        rgb[:,:,:,2] = blue
        return rgb
    
    def __add__(self, other: "Explanation[Sx,Sy]") -> "Explanation[Sx,Sy]":
        return self._compute(shap_values=self.shap_values + other.shap_values)
    
    def __sub__(self, other: "Explanation[Sx,Sy]") -> "Explanation[Sx,Sy]":
        return self._compute(shap_values=self.shap_values - other.shap_values)
    
    def __mul__(self, other: "Explanation[Sx,Sy]") -> "Explanation[Sx,Sy]":
        return self._compute(shap_values=self.shap_values * other.shap_values)
    
    def __truediv__(self, other: "Explanation[Sx,Sy]") -> "Explanation[Sx,Sy]":
        return self._compute(shap_values=self.shap_values / other.shap_values)

    def __matmul__(self, other: "Explanation[Sy,Sa]") -> "Explanation[Sx,Sa]":
        return self._compute(
            class_shape=other.class_shape,
            shap_values=self.shap_values @ other.shap_values
        )
    
    def __getitem__(self, indices: int|Sequence[int]|slice) -> NDArray[float64]:
        return self.shap_values[indices]
    
    @overload
    def _compute(self,
                 *,
                 shap_values:   NDArray[float64] = ...) -> "Explanation[Sx,Sy]": ...

    @overload
    def _compute(self,
                 *,
                 feature_shape: Sa, 
                 shap_values:   NDArray[float64] = ...) -> "Explanation[Sa,Sy]": ...
    
    @overload
    def _compute(self,
                 *,
                 class_shape:   Sb,
                 shap_values:   NDArray[float64] = ...) -> "Explanation[Sx,Sb]": ...

    @overload  
    def _compute(self,
                 *,
                 feature_shape: Sa, 
                 class_shape:   Sb,
                 shap_values:   NDArray[float64] = ...) -> "Explanation[Sa,Sb]": ...
    
    
    def _compute(self,
                 *,
                 feature_shape: Sa|None = None, 
                 class_shape:   Sb|None = None,
                 shap_values:   NDArray[float64]|None = None) -> "Explanation":
        return Explanation(
            feature_shape=self.feature_shape if feature_shape is None else feature_shape,
            class_shape=self.class_shape if class_shape is None else class_shape,
            compute_time=self.compute_time,
            shap_values=self.shap_values if shap_values is None else shap_values,
            base_values=self.base_values
        )
