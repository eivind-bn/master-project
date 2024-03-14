from . import *
from enum import Enum

class Action(NamedTuple):
    key_bind: str|None
    ale_id: int

class Actions(Action, Enum):
    NOOP = Action(None, 0)
    FIRE = Action(" ", 1)
    UP = Action("w", 2)
    RIGHT = Action("d", 3)
    LEFT = Action("a", 4)
    DOWN = Action("s", 5)
    UP_RIGHT = Action("wd", 6)
    UP_LEFT = Action("wa", 7)
    DOWN_RIGHT = Action("sd", 8)
    DOWN_LEFT = Action("sa", 9)
    UP_FIRE = Action("w ", 10)
    RIGHT_FIRE = Action("d ", 11)
    LEFT_FIRE = Action("a ", 12)
    DOWN_FIRE = Action("s ", 13)
    UP_RIGHT_FIRE = Action("wd ", 14)
    UP_LEFT_FIRE = Action("wa ", 15)
    DOWN_RIGHT_FIRE = Action("sd ", 16)
    DOWN_LEFT_FIRE = Action("sa ", 17)
