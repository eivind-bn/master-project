# %%
import torch

# %%
from typing import *
from typing import Any
from numpy.typing import NDArray
from enum import Enum
from dataclasses import dataclass

import numpy as np
import cv2

ResizeMode = Literal["normal", "autosize"]
RatioMode = Literal["free_ratio", "keep_ratio"]
StatusBarMode = Literal["normal", "expanded"]

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

EventType = Literal["push", "release", "toggle"]

T = TypeVar("T", covariant=True)



class KeyEvents(Generic[T], TypedDict, total=False):
    left_mouse: Callable[[],T]

class KeyboardEventMatcher(Generic[T], TypedDict, total=False):
    push: Dict[str,Callable[[],T]]|KeyEvents
    release: Dict[str,Callable[[],T]]
    toggle: Dict[str,Callable[[],T]]

class MouseButtonMatcher(Generic[T], TypedDict, total=False):
    left: Callable[[],T]
    right: Callable[[],T]
    middle: Callable[[],T]

class MouseMoveEvent(Protocol[T]):
    def __call__(self, y: int, x: int) -> T:
        ...

class MouseEventMatcher(Generic[T], TypedDict, total=False):
    push: MouseButtonMatcher[T]
    release: MouseButtonMatcher[T]
    toggle: MouseButtonMatcher[T]
    move: MouseMoveEvent[T]

class DeviceMatcher(Generic[T], TypedDict, total=False):
    keyboard: KeyboardEventMatcher[T]
    mouse: MouseEventMatcher[T]

class Event:
    
    def __init__(self, mouse_events: Iterable[MouseEvent], key_events: Iterable[int]) -> None:
        pass

    def match(self, matcher: DeviceMatcher[T]) -> T:
        return matcher["mouse"]["move"](1,2)

g = Event([],[]).match({
    "mouse": {
        "move": lambda a,b: 3,
        "push": {
            "left": lambda: 3
        }
    }
})



class WindowClosed(Exception):
    pass

class Window:

    def __init__(self, 
                 name: str,
                 fps: float|None                                        = None,
                 scale: float                                           = 1.0,
                 resize_mode: ResizeMode                                = "normal",
                 ratio_mode: RatioMode                                  = "keep_ratio",
                 statusbar_mode: StatusBarMode                          = "normal",
                 enabled: bool                                          = True
                 ) -> None:
        
        self._name = name
        self._delay = 1 if fps is None else int(1e3/fps)
        self._scale = scale
        self._enabled = enabled

        resize_flags: Dict[ResizeMode,int] = {
            "normal": cv2.WINDOW_NORMAL,
            "autosize": cv2.WINDOW_AUTOSIZE
        }
        ratio_flags: Dict[RatioMode,int] = {
            "free_ratio": cv2.WINDOW_FREERATIO,
            "keep_ratio": cv2.WINDOW_KEEPRATIO
        }
        statusbar_flags: Dict[StatusBarMode,int] = {
            "normal": cv2.WINDOW_GUI_NORMAL,
            "expanded": cv2.WINDOW_GUI_EXPANDED
        }

        self.resize_flag = resize_flags[resize_mode]
        self.ratio_flag = ratio_flags[ratio_mode]
        self.statusbar_flag = statusbar_flags[statusbar_mode]

        self.window_flag = self.resize_flag | self.ratio_flag | self.statusbar_flag
    
    def __enter__(self) -> Callable[[NDArray[np.uint8]],None]:
        if self._enabled:
            resized = False
            mouse_events: List[MouseEvent] = []
            
            def render(image: NDArray[np.uint8]) -> None:
                nonlocal resized
                if not resized:
                    resized = True
                    H,W = image.shape[:2]
                    cv2.resizeWindow(self._name, int(W*self._scale), int(H*self._scale))

                if cv2.getWindowProperty(self._name, cv2.WND_PROP_VISIBLE) < 1:
                    raise WindowClosed()

                cv2.imshow(self._name, image)
                key = cv2.waitKeyEx(self._delay)

                use_default = True

                int_cb = self.key_events.get(key, None)
                if int_cb is not None:
                    int_cb()
                    use_default = False

                try:
                    chr_cb = self.key_events.get(chr(key), None)
                    if chr_cb is not None:
                        chr_cb()
                        use_default = False
                except ValueError:
                    pass

                if use_default:
                    self.key_events.get(None, lambda: None)()

            cv2.namedWindow(self._name, self.window_flag)
            assert hasattr(cv2, "setMouseCallback")
            cv2.setMouseCallback(self._name, mouse_events.append)
            return render
        
        return lambda _: None

    
    def __exit__(self, *_) -> None:
        if self._enabled:
            cv2.destroyWindow(self._name)


# %%
import numpy as np
with Window("Asteroids") as window:
    for _ in range(1000):
        window(np.full((250,250), 127, dtype=np.uint8))

# %%



