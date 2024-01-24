from typing import *
from numpy.typing import NDArray
from .angle import Angle

import matplotlib.image as im
import matplotlib.pyplot as plt
import numpy as np

_PLAYER_COLOR = (240,128,128)
_Y_START, _Y_END = 18, 195

_CMAP = Literal[
            "viridis", "plasma", "inferno", "magma", "cividis",
            "Greys", "Purples", "Blues", "Greens", "Oranges", 
            "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", 
            "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
            "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
            "spring", "summer", "autumn", "winter", "cool", "Wistia",
            "hot", "afmhot", "gist_heat", "copper", "PiYG", "PRGn", 
            "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", 
            "coolwarm", "bwr", "seismic", "twilight", "twilight_shifted", 
            "hsv", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", 
            "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c",
            "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
            "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "gist_rainbow", 
            "rainbow", "jet", "turbo", "nipy_spectral", "gist_ncar"]


class Observation: 

    def __init__(self,
                 spaceship:         NDArray[np.uint8],
                 asteroids:         NDArray[np.uint8],
                 spaceship_angle:   Angle|None = None) -> None:
        self.spaceship = spaceship
        self.asteroids = asteroids
        self.spaceship_angle = spaceship_angle

    def numpy(self) -> NDArray[np.uint8]:
        return self.spaceship | self.asteroids
    
    def translated(self, new_center: Tuple[int,int]|None = None) -> "Observation":
        if new_center is None:
            new_center = self.find_player()

        if new_center is None:
            return self
        
        spaceship_copy = self.spaceship.copy()
        asteroids_copy = self.asteroids.copy()

        spaceship_view = spaceship_copy[_Y_START:_Y_END]
        asteroids_view = asteroids_copy[_Y_START:_Y_END]

        Y,X,_ = spaceship_view.shape    
        old_center = Y//2, X//2
        dy = old_center[0] - new_center[0]
        dx = old_center[1] - new_center[1]
        
        sy,sx,sc = np.nonzero(spaceship_view)
        ay,ax,ac = np.nonzero(asteroids_view)

        spaceship_view[sy,sx,sc] = 0
        asteroids_view[ay,ax,ac] = 0
        
        spaceship_view[(sy + dy)%Y,(sx + dx)%X,sc] = self.spaceship[_Y_START:_Y_END][sy,sx,sc]
        asteroids_view[(ay + dy)%Y,(ax + dx)%X,ac] = self.asteroids[_Y_START:_Y_END][ay,ax,ac]

        return Observation(
            spaceship=spaceship_copy,
            asteroids=asteroids_copy,
            spaceship_angle=self.spaceship_angle
        )

    def rotated(self) -> "Observation":
        # TODO: fix rendering issue when called twice

        def rotate_layer(layer: NDArray[np.uint8], angle: Angle) -> NDArray[np.uint8]:
            radians = -angle.radians()
            layer_copy = layer.copy()
            layer_view = layer_copy[_Y_START:_Y_END]

            Y,X = layer_view.shape[:2]
            y_center,x_center = Y//2, X//2

            idy,idx,idc = np.nonzero(layer_view)
            layer_view[idy,idx,idc] = 0

            y_ref = np.concatenate([idy-Y,idy,idy+Y]*3) - y_center
            x_ref = np.concatenate([idx-X]*3 + [idx]*3 + [idx+X]*3) - x_center

            y_rot = np.array(y_ref*radians.cos() - x_ref*radians.sin() + y_center)
            x_rot = np.array(y_ref*radians.sin() + x_ref*radians.cos() + x_center)

            y_rot_loc = np.array([
                y_rot - 0.5,
                y_rot,
                y_rot + 0.5
            ], dtype=int)

            x_rot_loc = np.array([
                x_rot - 0.5,
                x_rot,
                x_rot + 0.5
            ], dtype=int)

            within = (y_rot_loc >= 0) & (y_rot_loc < Y) & (x_rot_loc >= 0) & (x_rot_loc < X)

            for i in range(3):
                y_loc = (y_ref[within[i]] + y_center)%Y
                x_loc = (x_ref[within[i]] + x_center)%X

                layer_view[y_rot_loc[i][within[i]],x_rot_loc[i][within[i]]] =\
                    layer[_Y_START:_Y_END][y_loc,x_loc] 
                
            return layer_copy
        
        if self.spaceship_angle is None:
            return self
        else:
            return Observation(
                spaceship=rotate_layer(self.spaceship, angle=self.spaceship_angle),
                asteroids=rotate_layer(self.asteroids, angle=self.spaceship_angle)
            )
    
    def show(self, cmap: _CMAP|None = None) -> None:
        plt.imshow(self.numpy(), cmap=cmap)
        plt.show()

    def save(self, filename: str, cmap: _CMAP) -> None:
        im.imsave(fname=filename, arr=self.numpy(), cmap=cmap)

    def find_player(self, color: Tuple[int,int,int]|None = None) -> Tuple[int,int]|None:
        if color is None:
            color = _PLAYER_COLOR

        spaceship_view = self.spaceship[_Y_START:_Y_END]
        Y,X,_ = spaceship_view.shape

        player_indices = np.argwhere(np.all(spaceship_view == color, axis=2))

        if np.any(player_indices[:,0] > (Y*3)//4):
            player_indices[player_indices[:,0] <= Y//4,0] += Y

        if np.any(player_indices[:,1] > (X*3)//4):
            player_indices[player_indices[:,1] <= X//4,1] += X
        
        if player_indices.size > 0:
            py,px = tuple(int(point) for point in np.mean(player_indices, axis=0))
            return py,px
        
        return None
    