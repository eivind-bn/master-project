from . import *
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, float64

import shap # type: ignore
import numpy as np
import math

Ints: TypeAlias = Tuple[int,...]
Int: TypeAlias = Tuple[int]
C = TypeVar("C", bound=Ints)
F = TypeVar("F", bound=Ints)
C2 = TypeVar("C2", bound=Ints)
F2 = TypeVar("F2", bound=Ints)

class Explanation(Generic[C,F]):

    @overload
    def __init__(self,
                 *,
                 class_shape:   C, 
                 feature_shape: F, 
                 compute_time:  Seconds,
                 shap_values:   NDArray[float64]) -> None: ...
        
    @overload
    def __init__(self,
                 *,
                 class_shape:   C, 
                 feature_shape: F, 
                 explanation:   shap.Explanation) -> None: ...

    def __init__(self,
                 *,
                 class_shape:   C, 
                 feature_shape: F, 
                 compute_time:  Seconds|None = None,
                 shap_values:   NDArray[float64]|None = None,
                 explanation:   shap.Explanation|None = None) -> None:
        super().__init__()
        self._class_shape = class_shape
        self._feature_shape = feature_shape
        
        if compute_time is None or shap_values is None:
            if explanation is None:
                raise ValueError()
            
            self._shap_values: NDArray[float64] = explanation.values
            self._base_values: NDArray[float64] = explanation.base_values
            self._compute_time = Seconds(explanation.compute_time)

            assert self._base_values is not None

        else:
            if explanation is not None:
                raise ValueError()
                      
            self._shap_values = shap_values
            self._compute_time = compute_time

        shap_values_shape = self._class_shape + self._feature_shape
        if self._shap_values.shape != shap_values_shape:
            raise ValueError(f"Shap values are of invalid shape: {self._shap_values.shape}, expected: {shap_values_shape}")
        
        #if self._base_values.shape != self._class_shape:
        #    raise ValueError(f"Base values are of invalid shape: {self._base_values.shape}, expected: {self._class_shape}")

    @property
    def class_shape(self) -> C:
        return self._class_shape
    
    @property
    def class_items(self) -> int:
        return math.prod(self.class_shape)

    @property
    def feature_shape(self) -> F:
        return self._feature_shape
    
    @property
    def feature_items(self) -> int:
        return math.prod(self.feature_shape)
    
    @property
    def shape(self) -> Ints:
        return self.class_shape + self.feature_shape
    
    @property
    def items(self) -> int:
        return self.class_items + self.feature_items

    @property
    def shap_values(self) -> NDArray[float64]:
        return self._shap_values

    @property
    def compute_time(self) -> Seconds:
        return self._compute_time

    def flip(self) -> "Explanation[F,C]":
        flat_flipped = self.flatten().shap_values.T
        return self._compute(
            class_shape=self.feature_shape,
            feature_shape=self.class_shape,
            shap_values=flat_flipped.reshape(self.feature_shape + self.class_shape),
        )
    
    def reshape(self, 
                class_shape:    C2,
                feature_shape:  F2) -> "Explanation[C2,F2]":
        return self._compute(
            class_shape=class_shape,
            feature_shape=feature_shape,
            shap_values=self.shap_values.reshape(class_shape + feature_shape),
        )
    
    def flatten(self) -> "Explanation[Int,Int]":
        return self.reshape(
            class_shape=(math.prod(self.class_shape),),
            feature_shape=(math.prod(self.feature_shape),)
        )

    # def conjunct(self, other: "Explanation[C2,F]") -> "Explanation[C,C2]":
    #     self._assert_shape(other, feature_shape=self.feature_shape)
    #     A = self.flatten()
    #     B = other.flatten()
    #     return (A @ B.flip()).reshape(
    #         class_shape=self.class_shape,
    #         feature_shape=other.class_shape
    #     )

    def conjunct(self, other: "Explanation[C2,F2]") -> "Explanation[C2,F]":
        A = self.flatten()
        B = other.flatten()
        C = np.outer(B.shap_values, A.shap_values)\
            .reshape(B.shape + A.shape)\
            .sum((1,2))
        return Explanation(
            class_shape=other.class_shape,
            feature_shape=self.feature_shape,
            compute_time=self.compute_time + other.compute_time,
            shap_values=np.reshape(C, other.class_shape + self.feature_shape)
        )
    
    def attribution_weights(self) -> "Explanation[C,F]":
        flattened = self.flatten().shap_values
        weights: NDArray[float64] = np.abs(flattened)/np.sum(np.abs(flattened), axis=0)
        return Explanation(
            class_shape=self.class_shape,
            feature_shape=self.feature_shape,
            compute_time=self.compute_time,
            shap_values=weights.reshape(self.class_shape + self.feature_shape)
        )
    
    def contribution_weights(self) -> "Explanation[C,F]":
        flattened = self.flatten().shap_values
        weights: NDArray[float64] = flattened/np.sum(np.abs(flattened), axis=0)
        return Explanation(
            class_shape=self.class_shape,
            feature_shape=self.feature_shape,
            compute_time=self.compute_time,
            shap_values=weights.reshape(self.class_shape + self.feature_shape)
        )
    
    def eap(self, expander: "Explanation[C2,F2]") -> "Explanation[C,C2]":
        return expander.attribution_weights().flip().conjunct(self)
    
    def ecp(self, expander: "Explanation[C2,F2]") -> "Explanation[C,C2]":
        return expander.contribution_weights().flip().conjunct(self)
    
    def max(self) -> float:
        return float(self.shap_values.max())
    
    def abs(self) -> "Explanation[C,F]":
        return Explanation(
            class_shape=self.class_shape,
            feature_shape=self.feature_shape,
            compute_time=self.compute_time,
            shap_values=np.abs(self.shap_values),
        )
    
    def class_sum(self) -> "Explanation[Tuple[Literal[1]],F]":
        class_axis_cnt = len(self.class_shape)
        axes = tuple(range(class_axis_cnt))
        values: NDArray[float64] = self.shap_values.sum(axes)
        return Explanation(
            class_shape=(1,),
            feature_shape=self.feature_shape,
            compute_time=self.compute_time,
            shap_values=values.reshape((1,) + self.feature_shape),
        )
    
    def feature_sum(self) -> "Explanation[C,Tuple[Literal[1]]]":
        class_axis_cnt = len(self.class_shape)
        feature_axis_cnt = len(self.feature_shape)
        axes = tuple(range(class_axis_cnt, class_axis_cnt+feature_axis_cnt))
        values: NDArray[float64] = self.shap_values.sum(axes)
        return Explanation(
            class_shape=self.class_shape,
            feature_shape=(1,),
            compute_time=self.compute_time,
            shap_values=values.reshape(self.class_shape + (1,)),
        )
    
    def to_rgb(self: "Explanation[C,Tuple[int,int]]", norm: float|None = None) -> NDArray[float32]:
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
    
    def __add__(self, other: "Explanation[C,F]") -> "Explanation[C,F]":
        self._assert_shape(other, class_shape=self.class_shape, feature_shape=self.feature_shape)
        return self._compute(
            shap_values=self.shap_values + other.shap_values,
            )
    
    def __sub__(self, other: "Explanation[C,F]") -> "Explanation[C,F]":
        self._assert_shape(other, class_shape=self.class_shape, feature_shape=self.feature_shape)
        return self._compute(
            shap_values=self.shap_values - other.shap_values,
            )
    
    def __mul__(self, other: "Explanation[C,F]") -> "Explanation[C,F]":
        self._assert_shape(other, class_shape=self.class_shape, feature_shape=self.feature_shape)
        return self._compute(
            shap_values=self.shap_values * other.shap_values,
            )
    
    def __truediv__(self, other: "Explanation[C,F]") -> "Explanation[C,F]":
        self._assert_shape(other, class_shape=self.class_shape, feature_shape=self.feature_shape)
        return self._compute(
            shap_values=self.shap_values / other.shap_values,
            )

    def __matmul__(self, other: "Explanation[F,C2]") -> "Explanation[C,C2]":
        self._assert_shape(other, class_shape=self.feature_shape)
        return self._compute(
            feature_shape=other.feature_shape,
            shap_values=self.shap_values @ other.shap_values,
        )
    
    def __getitem__(self, indices: int|Sequence[int]|slice) -> NDArray[float64]:
        return self.shap_values[indices]
    
    @overload
    @staticmethod
    def _assert_shape(explanation:   "Explanation[C2,F2]",
                      *,
                      class_shape:   C2, 
                      feature_shape: F2) -> None: ...
            
    @overload
    @staticmethod
    def _assert_shape(explanation:   "Explanation[C2,F2]",
                      *,
                      class_shape:   C2) -> None: ...
    
    @overload
    @staticmethod
    def _assert_shape(explanation:   "Explanation[C2,F2]",
                      *,
                      feature_shape: F2) -> None: ...
    
    @staticmethod
    def _assert_shape(explanation:   "Explanation[C2,F2]",
                      *,
                      class_shape:   C2|None = None, 
                      feature_shape: F2|None = None) -> None:
        if class_shape is not None and class_shape != explanation.class_shape:
            raise ValueError(f"Class shape of: {class_shape} differs from expected shape: {explanation.class_shape}")
        if feature_shape is not None and feature_shape != explanation.feature_shape:
            raise ValueError(f"Feature shape of: {feature_shape} differs from expected shape: {explanation.feature_shape}")
    
    @overload
    def _compute(self,
                 *,
                 shap_values:   NDArray[float64] = ...) -> "Explanation[C,F]": ...

    @overload
    def _compute(self,
                 *,
                 feature_shape: F2, 
                 shap_values:   NDArray[float64] = ...) -> "Explanation[C,F2]": ...
    
    @overload
    def _compute(self,
                 *,
                 class_shape:   C2,
                 shap_values:   NDArray[float64] = ...) -> "Explanation[C2,F]": ...

    @overload  
    def _compute(self,
                 *,
                 class_shape:   C2,
                 feature_shape: F2, 
                 shap_values:   NDArray[float64] = ...) -> "Explanation[C2,F2]": ...
    
    
    def _compute(self,
                 *,
                 class_shape:   C2|None = None,
                 feature_shape: F2|None = None, 
                 shap_values:   NDArray[float64]|None = None) -> "Explanation":
        return Explanation(
            class_shape=self.class_shape if class_shape is None else class_shape,
            feature_shape=self.feature_shape if feature_shape is None else feature_shape,
            compute_time=self.compute_time,
            shap_values=self.shap_values if shap_values is None else shap_values,
        )
