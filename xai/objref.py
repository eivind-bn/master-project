from . import *
from abc import ABC, abstractmethod

import dill # type: ignore
import tempfile
import os

T = TypeVar("T")

class ObjRef(ABC, Generic[T]):

    @abstractmethod
    def size(self) -> Bytes:
        pass

    def dumped(self) -> "Dump[T]":
        with self as data:
            return Dump(data)

    def loaded(self) -> "Load[T]":
        with self as data:
            return Load(data)

    def filed(self, 
              directory: str|None = None, 
              prefix:    str|None = None, 
              suffix:    str|None = None) -> "File[T]":
        with self as data:
            return File(
                data=data,
                directory=directory,
                prefix=prefix,
                suffix=suffix
            )

    def save(self, path: str) -> None:
        with self as data, open(path, "w+b") as file:
            dill.dump(data, file)

    @abstractmethod
    def __enter__(self) -> T:
        pass

    @abstractmethod
    def __exit__(self, *_: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.size()})"
    
class File(ObjRef[T]):
    
    def __init__(self, 
                 data:      T, 
                 directory: str|None = None, 
                 prefix:    str|None = None, 
                 suffix:    str|None = None) -> None:
        
        super().__init__()
        self._enters = 0
        self._data: T|None = None

        if prefix is None:
            prefix = f"{data.__class__.__name__.lower()}_"

        with tempfile.NamedTemporaryFile(mode="w+b", dir=directory, prefix=prefix, suffix=suffix, delete=False) as file:
            dill.dump(data, file)
            self._size = Bytes(file.tell())
            self._name = file.name

    def size(self) -> Bytes:
        return self._size
        
    def __enter__(self) -> T:
        assert self._enters >= 0
        self._enters += 1
        if self._data is None:
            self._file = open(self._name, "r+b")
            self._file.seek(0)
            self._data = cast(T, dill.load(self._file))
        
        return self._data

    def __exit__(self, *_: Any) -> None:
        assert self._enters > 0
        self._enters -= 1
        if self._enters < 1:    
            self._file.seek(0)
            dill.dump(self._data, self._file)
            self._file.truncate()
            self._size = Bytes(self._file.tell())
            self._data = None
            self._file.close()

    def __del__(self) -> None:
        try:
            os.remove(self._name)
        except FileNotFoundError:
            pass

class Dump(ObjRef[T]):

    def __init__(self, data: T) -> None:
        super().__init__()
        self._enters = 0
        self._data: T|None = None
        self._file = tempfile.TemporaryFile(mode="w+b")
        dill.dump(data, self._file)
        self._size = Bytes(self._file.tell())

    def size(self) -> Bytes:
        return self._size

    def __enter__(self) -> T:
        assert self._enters >= 0
        self._enters += 1
        if self._data is None:
            self._file.seek(0)
            self._data = cast(T, dill.load(self._file))

        return self._data

    def __exit__(self, *_: Any) -> None:
        assert self._enters > 0
        self._enters -= 1
        if self._enters < 1:    
            self._file.seek(0)
            dill.dump(self._data, self._file)
            self._file.truncate()
            self._size = Bytes(self._file.tell())
            self._data = None

    def __del__(self) -> None:
        self._file.close()

class Load(ObjRef[T]):

    def __init__(self, data: T) -> None:
        super().__init__()
        self._enters = 0
        self._data: T|None = None
        self._obj_bytes = dill.dumps(data)

    def size(self) -> Bytes:
        if self._data is not None:
            self._obj_bytes = dill.dumps(self._data)

        return Bytes(len(self._obj_bytes))

    def __enter__(self) -> T:
        assert self._enters >= 0
        self._enters += 1
        if self._data is None:
            self._data = cast(T, dill.loads(self._obj_bytes))

        return self._data

    def __exit__(self, *_: Any) -> None:
        assert self._enters > 0
        self._enters = max(self._enters - 1, 0)
        if self._enters < 1 and self._data is not None:    
            self._obj_bytes = dill.dumps(self._data)
            self._data = None
