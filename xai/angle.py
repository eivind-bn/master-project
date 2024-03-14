from . import *

import math

class Angle(Unit["Angle"]):

    @classmethod
    def _convert_unit(cls, other: "Angle") -> int|float:
        return (other._value*cls.order)/other.order

    def sin(self) -> float:
        return math.sin(Radians._convert_unit(self))
    
    def cos(self) -> float:
        return math.cos(Radians._convert_unit(self))
    
    def tan(self) -> float:
        return math.tan(Radians._convert_unit(self))
    
    def turns(self) -> "Turns":
        return Turns.of(self)

    def radians(self) -> "Radians":
        return Radians.of(self)
    
    def degrees(self) -> "Degrees":
        return Degrees.of(self)
    
    def truncate(self) -> Self:
        return self.__class__(self._value % self.order)
    
    @classmethod
    def asin(cls, sine: float) -> Self:
        return cls.of(Radians(math.asin(sine)))
    
    @classmethod
    def acos(cls, cosine: float) -> Self:
        return cls.of(Radians(math.acos(cosine)))
    
    @classmethod
    def atan(cls, tangent: float) -> Self:
        return cls.of(Radians(math.atan(tangent)))
    
    def __eq__(self, operand: object) -> bool:
        if isinstance(operand, Angle):
            return self._value == self._convert_unit(operand)
        else:
            return False
    
class Turns(Angle):
    order = 1.0

class Radians(Angle):
    order = 2*math.pi

class Degrees(Angle):
    order = 360.0

