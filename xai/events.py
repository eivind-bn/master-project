from . import *
from enum import Enum
from dataclasses import dataclass

import cv2

T = TypeVar("T")

class MouseEventType(Enum):
    NONE                        = -1
    MOUSE_MOVE                  = cv2.EVENT_MOUSEMOVE
    LEFT_BUTTON_DOWN            = cv2.EVENT_LBUTTONDOWN
    RIGHT_BUTTON_DOWN           = cv2.EVENT_RBUTTONDOWN
    MIDDLE_BUTTON_DOWN          = cv2.EVENT_MBUTTONDOWN
    LEFT_BUTTON_UP              = cv2.EVENT_LBUTTONUP
    RIGHT_BUTTON_UP             = cv2.EVENT_RBUTTONUP
    MIDDLE_BUTTON_UP            = cv2.EVENT_MBUTTONUP
    LEFT_BUTTON_DOUBLE_BLCLK    = cv2.EVENT_LBUTTONDBLCLK
    RIGHT_BUTTON_DOUBLE_BLCLK   = cv2.EVENT_RBUTTONDBLCLK
    MIDDLE_BUTTON_DOUBLE_BLCLK  = cv2.EVENT_MBUTTONDBLCLK
    MOUSE_WHEEL                 = cv2.EVENT_MOUSEWHEEL
    MOUSEH_WHEEL                = cv2.EVENT_MOUSEHWHEEL

    @classmethod
    def _missing_(cls, _: Any) -> "MouseEventType":
        return cls.NONE

class MouseEventFlag(Enum):
    NONE            = -1
    LEFT_BUTTON     = cv2.EVENT_FLAG_LBUTTON
    RIGHT_BUTTON    = cv2.EVENT_FLAG_RBUTTON
    MIDDLE_BUTTON   = cv2.EVENT_FLAG_MBUTTON
    CTRL_KEY        = cv2.EVENT_FLAG_CTRLKEY
    SHIFT_KEY       = cv2.EVENT_FLAG_SHIFTKEY
    ALT_KEY         = cv2.EVENT_FLAG_ALTKEY

    @classmethod
    def _missing_(cls, _: Any) -> "MouseEventFlag":
        return cls.NONE

@dataclass
class MouseEvent:
    y: int
    x: int
    event: MouseEventType
    flags: MouseEventFlag


Case = Tuple[int|str,...]|str|None
Handler = Callable[[],T]

class Events:

    def __init__(self, 
                 key_events: Set[int], 
                 mouse_events: Iterable[MouseEvent]) -> None:
        self._key_events = tuple(sorted(key_events))
        self._mouse_events = tuple(mouse_events)
    
    def match(self, cases: Dict[Case,Handler[T]]) -> T:
        match_dict: Dict[Tuple[int,...]|None,Callable[[],T]] = {}
        
        for key,handler in cases.items():
            if isinstance(key, int):
                match_dict[key] = handler
            elif isinstance(key, str):
                match_dict[tuple(sorted(ord(char) for char in key))] = handler
            elif isinstance(key, tuple):
                key_codes: List[int] = []
                for sub_keys in key:
                    if isinstance(sub_keys, str):
                        for char in sub_keys:
                            key_codes.append(ord(char))
                    else:
                        key_codes.append(sub_keys)
                match_dict[tuple(sorted(key_codes))] = handler
            elif key is None:
                match_dict[key] = handler

        if self._key_events in match_dict:
            return match_dict[self._key_events]()
        elif None in match_dict:
            return match_dict[None]()
        else:
            raise KeyError()