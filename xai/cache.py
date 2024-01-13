from typing import *
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pickle
import tempfile

T = TypeVar("T")

class Cache(ABC, Generic[T]):

    @abstractmethod
    def dumped(self) -> "Dump[T]":
        pass

    @abstractmethod
    def loaded(self) -> "Load[T]":
        pass

    @abstractmethod
    def __enter__(self) -> T:
        pass

    @abstractmethod
    def __exit__(self, *_: Any) -> None:
        pass

class Dump(Cache[T]):

    def __init__(self, data: T) -> None:
        super().__init__()

        self._enters = 0
        self._data: T|None = None
        self._file = tempfile.TemporaryFile(mode="w+b")
        pickle.dump(data, self._file)

    def dumped(self) -> "Dump[T]":
        with self as data:
            return Dump(data=data)
        
    def loaded(self) -> "Load[T]":
        with self as data:
            return Load(data=data)

    def __enter__(self) -> T:
        assert self._enters >= 0
        self._enters += 1
        if self._data is None:
            self._file.seek(0)
            data: T = pickle.load(self._file)
            self._data = data
            return data
        else:
            return self._data

    def __exit__(self, *_: Any) -> None:
        assert self._enters > 0
        self._enters -= 1
        if self._enters < 1:    
            self._file.seek(0)
            pickle.dump(self._data, self._file)
            self._file.truncate()
            self._data = None

    def __del__(self) -> None:
        self._file.close()

class Load(Cache[T]):

    def __init__(self, data: T) -> None:
        super().__init__()
        self._data = data

    def dumped(self) -> Dump[T]:
        return Dump(self._data)
    
    def loaded(self) -> "Load[T]":
        return self

    def __enter__(self) -> T:
        return self._data

    def __exit__(self, *_: Any) -> None:
        pass
