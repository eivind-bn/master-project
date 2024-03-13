from typing import *
from numpy.typing import NDArray
from numpy import uint8
from . import *

import numpy as np
import cv2

ResizeMode = Literal["normal", "autosize"]
RatioMode = Literal["free_ratio", "keep_ratio"]
StatusBarMode = Literal["normal", "expanded"]

class WindowClosed(Exception):
    pass

class WindowInterface:

    def __init__(self, 
                 updater:   Callable[[NDArray[uint8]|None],Events]) -> None:
        self._updater = updater

    def update(self, image: NDArray[uint8]|None) -> Events:
        return self._updater(image)
    
    def break_window(self) -> NoReturn:
        raise WindowClosed()
    
    def __call__(self, image: NDArray[uint8]|None) -> Events:
        return self.update(image)


class Window:

    def __init__(self, 
                 name: str,
                 fps: float|None                = None,
                 scale: float                   = 1.0,
                 resize_mode: ResizeMode        = "normal",
                 ratio_mode: RatioMode          = "keep_ratio",
                 statusbar_mode: StatusBarMode  = "normal",
                 flip_color_endianness: bool    = True,
                 enabled: bool                  = True) -> None:
        
        self._name = name
        self._delay = MilliSeconds(1) if fps is None else Seconds(1)/fps
        self._next_end = MilliSeconds.now() + self._delay
        self._scale = scale
        self._enabled = enabled
        self._flip_color_endianness = flip_color_endianness
        self._resized = False
        self._mouse_events: List[MouseEvent] = []

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

    def _append_mouse_event(self, event_type: int, x: int, y: int, flags: int, *_: Any) -> None:
        self._mouse_events.append(MouseEvent(
            y=y,
            x=x,
            event=MouseEventType(event_type),
            flags=MouseEventFlag(flags)
        ))

    def _window_visible(self) -> bool:
        return cv2.getWindowProperty(self._name, cv2.WND_PROP_VISIBLE) > 0

    def _update_and_poll_events(self, image: NDArray[np.uint8]|None = None) -> Events:
        end = self._next_end

        if not self._window_visible():
            raise WindowClosed()
        
        if image is not None:
            cv2.imshow(self._name, image[:,:,::-1] if self._flip_color_endianness else image)

            if not self._resized:
                self._resized = True
                H,W = image.shape[:2]
                cv2.resizeWindow(self._name, int(W*self._scale), int(H*self._scale))

        key_events: Set[int] = set()

        now = MilliSeconds.now()
        key = cv2.waitKeyEx(max((end - now).int(), 1))
        if key != -1:
            key_events.add(key)

        now = MilliSeconds.now()
        while now < end:
            key = cv2.waitKeyEx(max((end - now).int(), 1))
            if key != -1:
                key_events.add(key)

            now = MilliSeconds.now()

        self._next_end = MilliSeconds.now() + self._delay
        mouse_events = self._mouse_events.copy()
        self._mouse_events.clear()

        return Events(
            key_events=key_events,
            mouse_events=mouse_events
        )
    
    def __enter__(self) -> WindowInterface:
        if self._enabled:
            cv2.namedWindow(self._name, self.window_flag)
            cv2.setMouseCallback(self._name, self._append_mouse_event)
            return WindowInterface(updater=self._update_and_poll_events)
        
        return WindowInterface(updater = lambda _: Events(set(),tuple())) 
    
    def __exit__(self, *_: Any) -> None:
        if self._enabled:
            cv2.destroyWindow(self._name)
