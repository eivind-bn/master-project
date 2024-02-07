from typing import *
from abc import ABC, abstractmethod
from .bytes import Bytes

import pickle
import tempfile

T = TypeVar("T")

class Cache(ABC, Generic[T]):

    @abstractmethod
    def size(self) -> Bytes:
        pass

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.size()})"

class Dump(Cache[T]):

    def __init__(self, data: T) -> None:
        super().__init__()

        self._enters = 0
        self._data: T|None = None
        self._file = tempfile.TemporaryFile(mode="w+b")
        pickle.dump(data, self._file)
        self._size = Bytes(self._file.tell())

    def size(self) -> Bytes:
        return self._size

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
            self._size = Bytes(self._file.tell())
            self._data = None

    def __del__(self) -> None:
        self._file.close()

class Load(Cache[T]):

    def __init__(self, data: T) -> None:
        super().__init__()
        self._data = data
        self._size: Bytes|None = None

    def size(self) -> Bytes:
        if self._size is None:
            self._size = Bytes(len(pickle.dumps(self._data)))
        return self._size

    def dumped(self) -> Dump[T]:
        return Dump(self._data)
    
    def loaded(self) -> "Load[T]":
        return self

    def __enter__(self) -> T:
        self._size = None
        return self._data

    def __exit__(self, *_: Any) -> None:
        pass
