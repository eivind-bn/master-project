from typing import *
from ale_py import ALEInterface
from ale_py.roms import Asteroids as AsteroidsROM
from random import random
from numpy import uint8
from numpy.typing import NDArray
from action import Action
from observation import Observation
from reward import Reward
from response import Response

import cv2

class Asteroids:

    def __init__(self) -> None:
        
        self._emulator: ALEInterface = ALEInterface()
        self._emulator.loadROM(AsteroidsROM)
        self.reset()

        self._angle_steps_to_radians = (
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
        assert (sum(self._angle_steps_to_radians) - 47.24187379318632) < 1e-4
        
        height, width = self._emulator.getScreenDims()
        self.observation_shape = (height, width, 3)
        

    def step(self, 
             action:        "Action",
             stochastic:    bool = True,
             steps:         int = 1) -> Response:
        
        responses = 

        for _ in range(steps):
            if stochastic:
                if random() < 0.5:
                    spaceship, reward = self._step_spaceship(Action.NOOP)
                else:
                    asteroids, reward = self._step_asteroids(Action.NOOP)

            reward += self._step_spaceship(action)
            reward += self._step_asteroids(action)

            native_rewards.append(reward)
            images.append(Observation(
                spaceship=self.spaceship,
                asteroids=self.asteroids,
                spaceship_angle=self.spaceship_angle()
            ))

        observation = Observation(
            spaceship=self.spaceship, 
            asteroids=self.asteroids,
            spaceship_angle=self.spaceship_angle()
            )
        
        reward = Reward(
            values=native_rewards,
            observations=images,
            actions=[action]*steps
            )
        
        return observation, reward
    
    def _step_asteroids(self, action: Action) -> Tuple[NDArray[uint8],int]:
        flags = int(self._emulator.getRAM()[57])
        self._emulator.setRAM(57,~1&flags)
        reward = self._emulator.act(action.value)
        asteroids = self._emulator.getScreenRGB()
        return asteroids, reward
    
    def _step_spaceship(self, action: Action) -> Tuple[NDArray[uint8],int]:
        flags = int(self._emulator.getRAM()[57])
        self._emulator.setRAM(57,1|flags)
        reward = self._emulator.act(action.value)
        spaceship = self._emulator.getScreenRGB()
        return spaceship, reward
        
    def running(self) -> bool:
        return not self._emulator.game_over()
    
    def lives(self) -> int:
        return self._emulator.lives()
    
    def spaceship_angle(self) -> float:
        angle_step = self._emulator.getRAM()[60] & 0xf
        return self._angle_steps_to_radians[angle_step]

    def reset(self) -> None:
        self._emulator.reset_game()

    def play()

    def play(self,
             fps:           int = 60,
             scale:         float = 4.0,
             translate:     bool = False,
             rotate:        bool = False,
             stochastic:    bool = False,
             step_cb:       Callable[[AsteroidsObservation,AsteroidsReward],None] = lambda *_: None) -> None:
        
        title = "Asteroids"

        def window_visible() -> bool:
            return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) > 0
        
        def wait_key() -> str|None:
            code = cv2.waitKeyEx(1000//fps)
            if code > -1:
                return chr(code)
            else:
                return None  

        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)

            h,w,_ = self.observation_shape
            cv2.resizeWindow(title, int(w*scale), int(h*scale))

            while self.running() and window_visible():
                
                image = self.render()

                if translate:
                    image = image.translated()

                if rotate:
                    image = image.rotated()
                
                cv2.imshow(title, image.numpy()[:,:,::-1])
                
                key = wait_key()

                match key:
                    case "q":
                        break
                    case "w":
                        step_cb(image, self.step(AsteroidsAction.UP, stochastic=stochastic))
                    case "a":
                        step_cb(image, self.step(AsteroidsAction.LEFT, stochastic=stochastic))
                    case "d":
                        step_cb(image, self.step(AsteroidsAction.RIGHT, stochastic=stochastic))
                    case " ":
                        step_cb(image, self.step(AsteroidsAction.FIRE, stochastic=stochastic))
                    case _:
                        step_cb(image, self.step(AsteroidsAction.NOOP, stochastic=stochastic))

        finally:
            cv2.destroyWindow(title)
