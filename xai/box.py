from typing import *

T = TypeVar("T")

class Box(Generic[T]):

    def __init__(self, value: T) -> None:
        super().__init__()
        self._value = value

    def get(self) -> T:
        return self._value
    
    def set(self, value: T) -> None:
        self._value = value
        