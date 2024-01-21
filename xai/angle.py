from typing import *
from abc import ABC, abstractmethod
from enum import Enum

import math

T = TypeVar("T", int, float)

class Angle(ABC, Generic[T]):

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value

    @abstractmethod
    def radians(self) -> "Radians":
        pass
    
    @abstractmethod
    def degrees(self) -> "Degrees":
        pass

class Radians(Angle[float]):
    _max_radians = 2*math.pi

    def __init__(self, radians: float) -> None:
        super().__init__(radians % Radians._max_radians)

    def radians(self) -> "Radians":
        return self.value

    def degrees(self) -> "Degrees":
        return Radians(
            radians=(self.value*Degrees._max_degrees)/Radians._max_radians
        )

class Degrees(Angle[float]):
    _max_degrees = 360.0

    def __init__(self, degrees: float) -> None:
        super().__init__(degrees % Degrees._max_degrees)

    def radians(self) -> "Radians":
        return Radians(
            radians=(self.value*Radians._max_radians)/Degrees._max_degrees
        )
    
    def degrees(self) -> "Degrees":
        return self

class AngleState(Angle[int]):
    _state_to_angles: Tuple[Tuple["Radians","Degrees"],...]

    def __init__(self, state: int) -> None:
        super().__init__(state)
        assert state < len(AngleState._state_to_angles)

    def radians(self) -> "Radians":
        return AngleState._state_to_angles[self.value]
    
    def degrees(self) -> "Degrees":
        return self.radians().degrees()

_state_to_radian_values = (
    0.0, 
    0.23186466084938862, 
    0.5880026035475675, 
    0.9037239459029813, 
    1.5707963267948966, 
    2.256525837701183, 
    2.6909313275091598, 
    2.936197264400026, 
    3.141592653589793, 
    3.2834897081939567, 
    3.597664649939404, 
    4.023464592169828, 
    4.71238898038469, 
    5.365235611485464, 
    5.81953769817878, 
    6.120457932539206
    )
assert (sum(_state_to_radian_values) - 47.24187379318632) < 1e-4

state_angles: List[Tuple["Radians","Degrees"]] = []

for radian_value in _state_to_radian_values:
    radian = Radians(radians=radian_value)
    degree = radian.degrees()
    state_angles.append((radian, degree))

AngleState._state_to_angles = tuple(state_angles)

del _state_to_radian_values
