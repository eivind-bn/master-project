from typing import *
from abc import ABC, abstractmethod

X = TypeVar("X", bound="Unit")

class Unit(ABC, Generic[X]):
    order = 1e0

    def __init__(self, value: int | float) -> None:
        super().__init__()
        self._value = value

    @classmethod
    def of(cls, other: X) -> Self:
        return cls(cls._convert_unit(other))
    
    @classmethod
    def _convert_unit(cls, other: X) -> int|float:
        return (other._value*other.order)/cls.order

    def __add__(self, operand: X) -> Self:
        return self.__class__(self._value + self._convert_unit(operand))
    
    def __sub__(self, operand: X) -> Self:
        return self.__class__(self._value - self._convert_unit(operand))
    
    def __mul__(self, operand: int|float) -> Self:
        return self.__class__(self._value*operand)
    
    def __pow__(self, operand: int|float) -> Self:
        return self.__class__(self._value**operand)
    
    def __truediv__(self, operand: int|float) -> Self:
        return self.__class__(self._value/operand)
    
    def __floordiv__(self, operand: int|float) -> Self:
        return self.__class__(self._value//operand)
    
    def __mod__(self, operand: X) -> Self:
        return self.__class__(self._value % self._convert_unit(operand))
    
    def __neg__(self) -> Self:
        return self.__class__(-self._value)
    
    def __iadd__(self, operand: X) -> Self:
        self._value += self._convert_unit(operand)
        return self
    
    def __isub__(self, operand: X) -> Self:
        self._value -= self._convert_unit(operand)
        return self
    
    def __imul__(self, operand: int|float) -> Self:
        self._value *= operand
        return self
    
    def __ipow__(self, operand: int|float) -> Self:
        self._value **= operand
        return self
    
    def __itruediv__(self, operand: int|float) -> Self:
        self._value /= operand
        return self
    
    def __ifloordiv__(self, operand: int|float) -> Self:
        self._value //= operand
        return self
    
    def __imod__(self, operand: X) -> Self:
        self._value %= self._convert_unit(operand)
        return self
    
    @abstractmethod
    def __eq__(self, operand: object) -> bool:
        pass
    
    def __ne__(self, operand: object) -> bool:
        return not self.__eq__(operand)
    
    def __lt__(self, operand: X) -> bool:
        return self._value < self._convert_unit(operand)
    
    def __le__(self, operand: X) -> bool:
        return self._value <= self._convert_unit(operand)
    
    def __gt__(self, operand: X) -> bool:
        return self._value > self._convert_unit(operand)
    
    def __ge__(self, operand: X) -> bool:
        return self._value >= self._convert_unit(operand)
    
    def __round__(self, number_of_digits: int|None = None) -> Self:
        return self.__class__(round(self._value, number_of_digits))
    
    def __int__(self) -> int:
        return int(self._value)
    
    def __float__(self) -> float:
        return float(self._value)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"
    
    def int(self) -> int:
        return int(self._value)
    
    def float(self) -> float:
        return float(self._value)

class Exa(Unit[X]):
    order = 1e18

class Peta(Unit[X]):
    order = 1e15

class Tera(Unit[X]):
    order = 1e12

class Giga(Unit[X]):
    order = 1e9

class Mega(Unit[X]):
    order = 1e6

class Kilo(Unit[X]):
    order = 1e3

class Hecto(Unit[X]):
    order = 1e2

class Deca(Unit[X]):
    order = 1e1

class Deci(Unit[X]):
    order = 1e-1

class Centi(Unit[X]):
    order = 1e-2

class Milli(Unit[X]):
    order = 1e-3

class Micro(Unit[X]):
    order = 1e-6

class Nano(Unit[X]):
    order = 1e-9
