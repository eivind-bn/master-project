from typing import *
from numpy.typing import NDArray, DTypeLike
from collections import deque

from . import *

import numpy as np
import math
import pickle

T = TypeVar("T")

Shape: TypeAlias = Tuple[int,...]
DataType: TypeAlias = Literal[
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint128",
    "uint256",

    "int8",
    "int16",
    "int32",
    "int64",
    "int128",
    "int256",

    "float16",
    "float32",
    "float64",
    "float80",
    "float96",
    "float128",
    "float256",
]

class ArrayBuffer:

    def __init__(self, 
                 *,
                 data:          Mapping[str,NDArray]|None = None,
                 capacity:      int|Memory, 
                 schema:        Mapping[str, Tuple[Shape|int, DataType]]) -> None:
        
        self._schema: Mapping[str, Tuple[Shape, DataType]] = {}

        for name,(shape,dtype) in schema.items():
            if isinstance(shape, int):
                shape = (shape,)
            self._schema[name] = (shape, dtype)

        if isinstance(capacity, int):
            self._row_capacity = capacity
        else:
            row_size = 0
            for shape,dtype in self._schema.values():
                row_size += math.prod(shape) * np.dtype(dtype).itemsize

            self._row_capacity = int(capacity.bytes().float() / row_size)

        self._length = 0
        self._next_row = 0
        self._mem_capacity = Bytes(0)
        self._arrays: Dict[str,NDArray] = {}

        for row_name,(shape,dtype) in self._schema.items():
            array = np.zeros((self._row_capacity,) + shape, dtype=np.dtype(dtype))
            self._arrays[row_name] = array
            self._mem_capacity += Bytes(array.nbytes)

        if data is not None:
            self.append(data)

    def data(self, occupied_only: bool) -> Dict[str,NDArray]:
        if occupied_only:
            return {name:array[:self._length] for name,array in self._arrays.items()}
        else:
            return {name:array for name,array in self._arrays.items()}

    def append(self, rows: Mapping[str,NDArray]) -> None:
        if not rows:
            return
        
        row_cnt: int|None = None

        for row_name, dst in self.data(False).items():
            src = rows[row_name]
            if src.shape == dst.shape[1:]:
                if row_cnt is None:
                    row_cnt = 1
                elif row_cnt != 1:
                    raise ValueError(f"Expected {row_name} to contain {row_cnt} rows, but found 1")
            elif src.shape[1:] == dst.shape[1:]:
                if row_cnt is None:
                    row_cnt = src.shape[0]
                elif row_cnt != src.shape[0]:
                    raise ValueError(f"Expected {row_name} to contain {row_cnt} rows, but found {src.shape[0]}")  
            else:
                raise ValueError(f"{row_name} has incorrect shape {src.shape}. Expected {dst.shape[1:]}")
        
        assert row_cnt
        indices = np.arange(self._next_row, self._next_row + row_cnt) % self._row_capacity

        for row_name, dst in self.data(False).items():
            dst[indices] = rows[row_name]
        
        self._length = min(self._length + row_cnt, self._row_capacity)
        
    def resized(self, capacity: int|Memory, copy: bool) -> "ArrayBuffer":
        return ArrayBuffer(
            data=self.data(True).copy() if copy else self.data(True),
            capacity=capacity,
            schema=self._schema)

    def mini_batch(self, rows: int) -> Dict[str,NDArray]:
        occupied_rows = self.data(True)
        indices = np.arange(0, self._length)
        np.random.shuffle(indices)
        return {name:cast(NDArray, array[indices[:rows]]) for name,array in occupied_rows.items()}
    
    def save(self, path: str) -> None:
        with open(path, "w+b") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> "ArrayBuffer":
        with open(path, "rb") as file:
            buffer = pickle.load(file)
            if isinstance(buffer, cls):
                return buffer
            else:
                raise TypeError(f"Unpickled object is of incorrect type: {type(buffer)}")
        
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        dim = {name:array.shape[1:] for name,array in self.data(False).items()}
        length = len(self)
        max_length = self._row_capacity
        max_memsize = self._mem_capacity.gigabytes().float()
        memsize = (length/max_length)*max_memsize
        return f"{cls_name}({dim=}, rows={length}/{max_length}, memsize={memsize:.2f}/{max_memsize:.2f}GB)"

    def __len__(self) -> int:
        return self._length