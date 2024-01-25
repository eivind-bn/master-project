from typing import *
from ale_py import ALEInterface, LoggerMode
from ale_py.roms import Asteroids as AsteroidsROM # type: ignore[attr-defined]
from random import random
from numpy import uint8
from numpy.typing import NDArray
from .action import Actions, Action
from .observation import Observation
from .reward import Reward
from .window import Window, WindowClosed
from .record import Recorder
from .angle import Radians
from .angle_state import AngleStates

ALEInterface.setLoggerMode(LoggerMode.Warning)

class Asteroids:

    def __init__(self) -> None:
        
        self._emulator: ALEInterface = ALEInterface()
        self._emulator.loadROM(AsteroidsROM)
        
        height, width = self._emulator.getScreenDims()
        self.observation_shape = cast(Tuple[int,int,int], (height, width, 3))
        self._angle_states = tuple(AngleStates)

    def step(self, 
             action:        "Action",
             stochastic:    bool = True,
             steps:         int = 1) -> Tuple[Observation, Tuple[Reward,...]]:
        
        rewards: List[Reward] = []

        for _ in range(steps):
            if stochastic:
                if random() < 0.5:
                    _, reward = self._step_spaceship(Actions.NOOP)
                else:
                    _, reward = self._step_asteroids(Actions.NOOP)

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
        
        return observation, tuple(rewards)
    
    def _step_asteroids(self, action: Action) -> Tuple[NDArray[uint8],int]:
        flags = int(self._emulator.getRAM()[57])
        self._emulator.setRAM(57,~1&flags)
        reward = self._emulator.act(action.ale_id)
        asteroids = self._emulator.getScreenRGB()
        return asteroids, reward
    
    def _step_spaceship(self, action: Action) -> Tuple[NDArray[uint8],int]:
        flags = int(self._emulator.getRAM()[57])
        self._emulator.setRAM(57,1|flags)
        reward = self._emulator.act(action.ale_id)
        spaceship = self._emulator.getScreenRGB()
        return spaceship, reward
        
    def running(self) -> bool:
        return not self._emulator.game_over()
    
    def lives(self) -> int:
        return self._emulator.lives()
    
    def spaceship_angle(self) -> Radians:
        angle_step = int(self._emulator.getRAM()[60] & 0xf)
        return self._angle_states[angle_step]

    def reset(self) -> Tuple[Observation, Tuple[Reward,...]]:
        self._emulator.reset_game()
        return self.step(Actions.NOOP)

    def play(self,
             fps:           int = 60,
             scale:         float = 4.0,
             translate:     bool = False,
             rotate:        bool = False,
             stochastic:    bool = False,
             show:          bool = False,
             record_path:   str|None = None) -> None:
        
        observation, rewards = self.reset()

        def step(action: Action) -> Callable[[], Tuple[Observation, Tuple[Reward,...]]]:
            return lambda: self.step(action, stochastic)
        
        with Window(name="Asteroids", enabled=show, fps=fps, scale=scale) as window:
            with Recorder(filename=record_path, fps=fps, scale=scale) as recorder:
                cases = {
                    **{action.key_bind:step(action) for action in Actions},
                    "q": lambda: window.break_window(),
                    }
                try:
                    while self.running():

                        if translate:
                            observation = observation.translated()
                        if rotate:
                            observation = observation.rotated()

                        image = observation.numpy()
                        recorder(image)
                        observation, rewards = window(image).match(cases)
                except WindowClosed:
                    pass



