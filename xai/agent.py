from typing import *
from abc import ABC, abstractmethod
from .observation import Observation
from .action import Action
from .window import Window
from .record import Recorder
from .asteroids import Asteroids

import dill
import copy

class Agent(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.stats: Dict[str,Any] = {}

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            dill.dump(self, file)

    def clone(self) -> Self:
        return copy.deepcopy(self)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as file:
            dqn: Self = dill.load(file)

        if isinstance(dqn, cls):
            return dqn
        else:
            raise TypeError(f"Invalid type: {type(dqn)}")
        
    def play(self, rounds: int|None = None, record_path: str|None = None, scale: float = 4.0) -> None:
        with Window("Asteroids", fps=60, scale=scale) as window:
            with Recorder(filename=record_path, scale=scale) as recorder:
                env = Asteroids()
                if rounds is None:
                    while True:
                        observation,rewards = env.reset()
                        while env.running():
                            actions = self.predict([observation])
                            for action in actions:
                                observation,rewards = env.step(action)
                                image = observation.numpy(normalize=False)
                                recorder(image)
                                window.update(image).match({
                                    "q": lambda: window.break_window(),
                                    None: lambda: None
                                })
                else:
                    for round in range(rounds):
                        observation,rewards = env.reset()
                        while env.running():
                            actions = self.predict([observation])
                            for action in actions:
                                observation,rewards = env.step(action)
                                image = observation.numpy(normalize=False)
                                recorder(image)
                                window.update(image).match({
                                    "q": lambda: window.break_window(),
                                    None: lambda: None
                                })


        