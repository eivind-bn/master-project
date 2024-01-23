from typing import *
from ale_py import ALEInterface, LoggerMode
from ale_py.roms import Asteroids as AsteroidsROM
from random import random
from numpy import uint8
from numpy.typing import NDArray
from .action import Action
from .observation import Observation
from .reward import Reward
from .step import Step
from .window import Window
from .record import Recorder
from .angle import Radians

ALEInterface.setLoggerMode(LoggerMode.Warning)

class Asteroids:

    def __init__(self) -> None:
        
        self._emulator: ALEInterface = ALEInterface()
        self._emulator.loadROM(AsteroidsROM)

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
             steps:         int = 1) -> Step:
        
        rewards: List[Reward] = []

        for _ in range(steps):
            if stochastic:
                if random() < 0.5:
                    _, reward = self._step_spaceship(Action.NOOP)
                else:
                    _, reward = self._step_asteroids(Action.NOOP)

                rewards.append(Reward(reward))

            self.spaceship, reward = self._step_spaceship(action)
            rewards.append(Reward(reward))

            self.asteroids, reward = self._step_asteroids(action)
            rewards.append(Reward(reward))

        observation = Observation(
            spaceship=self.spaceship,
            asteroids=self.asteroids,
            spaceship_angle=self.spaceship_angle()
            )
        
        return Step(
            observation=observation,
            rewards=rewards
            )
    
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
    
    def spaceship_angle(self) -> Radians:
        angle_step = self._emulator.getRAM()[60] & 0xf
        return Radians(self._angle_steps_to_radians[angle_step])

    def reset(self) -> Step:
        self._emulator.reset_game()
        return self.step(Action.NOOP)

    def play(self,
             fps:           int = 60,
             scale:         float = 4.0,
             translate:     bool = False,
             rotate:        bool = False,
             stochastic:    bool = False,
             show:          bool = False,
             record_path:   str|None = None) -> None:
        
        step = self.reset()
        
        controller = {
            "w": Action.UP,
            "a": Action.LEFT,
            "d": Action.RIGHT,
            " ": Action.FIRE
        }
        with Window(name="Asteroids", enabled=show, fps=fps, scale=scale, key_events=controller) as window:
            with Recorder(filename=record_path, fps=fps, scale=scale) as recorder:
                self.reset()
                while self.running():
                    observation, reward = self.step(Action.NOOP)
                    window()
                    window().match({
                        "mouse": {
                            "click": {
                                
                            }
                        }
                    })


