from typing import *
from numpy.typing import NDArray
from skvideo.io import FFmpegWriter # type: ignore[import-untyped]

import cv2
import numpy as np

InterpolationMode = Literal[
    "nearest", 
    "nearest-exact", 
    "linear", 
    "linear-exact",
    "cubic", 
    "area", 
    "lanczos"]

class Recorder:

    def __init__(self, 
                 filename: str|None                 = None,
                 interpolation: InterpolationMode   = "nearest",
                 fps: int                           = 60,
                 scale: float                       = 1.0
                 ) -> None:
        
        self._filename = filename
        self._fps = fps
        self._scale = scale
        self._writer: FFmpegWriter|None = None

        interpolation_flags: Dict[InterpolationMode,int] = {
            "nearest": cv2.INTER_NEAREST,
            "nearest-exact": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "linear-exact": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4
        }

        self.interpolation_flag = interpolation_flags[interpolation]
        

    def __enter__(self) -> Callable[[NDArray[np.uint8]],None]:
        if self._filename is not None:
            writer = FFmpegWriter(
                filename=self._filename,
                inputdict={'-r': str(self._fps)},
                outputdict={'-r': str(self._fps)}
                )
            dim: Tuple[int,int]|None = None

            def render(image: NDArray[np.uint8]) -> None:
                nonlocal dim
                if dim is None:
                    H,W,*_ = tuple(int(dim*self._scale) for dim in image.shape[:2])
                    dim = (H,W)

                rescale = cv2.resize(
                    src=image, 
                    dsize=(dim[1], dim[0]),
                    interpolation=self.interpolation_flag)

                writer.writeFrame(rescale)
                
            self._writer = writer
            return render
        
        return lambda _: None
    
    def __exit__(self, *_: Any) -> None:
        if self._writer is not None:
            self._writer.close()
