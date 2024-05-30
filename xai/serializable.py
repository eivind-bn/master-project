from . import *

import dill # type: ignore

T = TypeVar("T")

class Serializable(Generic[T]):

    def save(self, path: str) -> None:
        with open(path, "w+b") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as file:
            obj: Serializable = dill.load(file)

        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError(f"Unpickled object is of incorrect type: {type(obj)}")