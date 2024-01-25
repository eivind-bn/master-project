
from typing import *
from enum import Enum
from .angle import Radians

AngleStateMeta: Any = type("AngleStateMeta", (type(Enum), type(Radians)), {})

class AngleStates(Radians, Enum, metaclass=AngleStateMeta): # type: ignore
    NNNN = 0.0, 
    NNNW = 0.23186466084938862, 
    NNWW = 0.5880026035475675, 
    NWWW = 0.9037239459029813, 
    WWWW = 1.5707963267948966, 
    WWWS = 2.256525837701183, 
    WWSS = 2.6909313275091598, 
    WSSS = 2.936197264400026, 
    SSSS = 3.141592653589793, 
    SSSE = 3.2834897081939567, 
    SSEE = 3.597664649939404, 
    SEEE = 4.023464592169828, 
    EEEE = 4.71238898038469, 
    EEEN = 5.365235611485464, 
    EENN = 5.81953769817878, 
    ENNN = 6.120457932539206

    @classmethod
    def __iter__(cls) -> Iterator[Radians]:
        return iter(cls)

assert (sum(angle.float() for angle in iter(AngleStates)) - 47.24187379318632) < 1e-4