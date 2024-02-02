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

    def get_and_set(self, value: T) -> T:
        old_value = self.get()
        self.set(value)
        return old_value
    
    def set_and_get(self, value: T) -> T:
        self.set(value)
        return self.get()
        