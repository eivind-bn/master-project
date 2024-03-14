from . import *

X = TypeVar("X", covariant=True)
Y = TypeVar("Y", covariant=True)

class Lazy(Generic[X]):

    def __init__(self, get_value: "Callable[[],Lazy[X]|X]") -> None:
        super().__init__()
        self._get_value = get_value
        self._value: X|None = None

    def map(self, f: "Callable[[X],Lazy[Y]|Y]") -> "Lazy[Y]":
        return Lazy(lambda: f(self()))

    def __call__(self) -> X:
        if self._value is None:
            value = self._get_value()
            if isinstance(value, Lazy):
                self._value = value()
            else:
                self._value = value

        return self._value
    
    def __repr__(self) -> str:
        return str(self())