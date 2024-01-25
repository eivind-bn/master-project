from typing import *
from enum import Enum
from dataclasses import dataclass
from .angle import Radians

@dataclass(frozen=True)
class Action:
    key_bind: str|None
    ale_id: int

class Actions(Action, Enum):
    NOOP = None, 0
    FIRE = " ", 1
    UP = "w", 2
    RIGHT = "d", 3
    LEFT = "a", 4
    DOWN = "s", 5
    UP_RIGHT = "wd", 6
    UP_LEFT = "wa", 7
    DOWN_RIGHT = "sd", 8
    DOWN_LEFT = "sa", 9
    UP_FIRE = "w ", 10
    RIGHT_FIRE = "d ", 11
    LEFT_FIRE = "a ", 12
    DOWN_FIRE = "s ", 13
    UP_RIGHT_FIRE = "wd ", 14
    UP_LEFT_FIRE = "wa ", 15
    DOWN_RIGHT_FIRE = "sd ", 16
    DOWN_LEFT_FIRE = "sa ", 17
