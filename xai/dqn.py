from typing import *
from collections import deque
from numpy.typing import NDArray
from numpy import uint8
from torch import Tensor
from tqdm import tqdm
from .policy import Device, Policy
from .action import Actions, Action
from .asteroids import Asteroids
from .window import Window, WindowClosed
from .record import Recorder
from .observation import Observation
from .explainer import Explainer
from .reward import Reward
from .feed_forward import FeedForward
from .step import Step
from .optimizer import Adam
from .agent import Agent
from .stream import Stream
from .reflist import RefList
from .bytes import GigaBytes, Memory
from .buffer import ArrayBuffer

import torch
import random
import numpy as np
import copy

class DQN(Agent):

    def __init__(self, device: Device, translate: bool, rotate: bool, replay_buffer_size: int|Memory) -> None:
        super().__init__()
        self._translate = translate
        self._rotate = rotate
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self._actions = (
            Actions.NOOP,
            Actions.UP,
            Actions.LEFT,
            Actions.RIGHT,
            Actions.FIRE,
        )
        self._policy = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), stride=(1,1)), # (102, 76)
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5,5), stride=(2,2)), # (107, 76)
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7,7), stride=(3,3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7,7), stride=(4,4)),
            torch.nn.ReLU(),

            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(in_features=64*7*5, out_features=32*7*5),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32*7*5, out_features=16*7*5),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16*7*5, out_features=len(self._actions)),
        ).to(device=self._device)

        self._replay_buffer = ArrayBuffer(
            capacity=replay_buffer_size,
            schema={
                "state":        ((210,160,3), "float32"),
                "action":       (len(self._actions), "int16"),
                "reward":       (1, "float32"),
                "next_state":   ((210,160,3), "float32"),
                "done":         (1, "float32")
            }
        )

    @overload
    def state(self, observation: Observation, to_tensor: Literal[False]) -> NDArray[np.float32]: ...

    @overload
    def state(self, observation: Observation, to_tensor: Literal[True]) -> Tensor: ...
        
    def state(self, observation: Observation, to_tensor: bool) -> NDArray[np.float32]|Tensor:
        if self._translate:
            observation = observation.translated()

        if self._rotate:
            observation = observation.rotated()

        if to_tensor:
            return observation.tensor(normalize=True, device=self._device)
        else:
            return observation.numpy(normalize=True)

    @overload 
    def predict(self, 
                observations: Sequence[Observation], 
                *, 
                to_action: Literal[True] = True) -> Sequence[Action]: ...

    @overload 
    def predict(self, 
                observations: Sequence[Observation], 
                *,
                to_action: Literal[False] = ...) -> Tensor: ...

    def predict(self, 
                observations: Sequence[Observation], 
                *,
                to_action: bool = True) -> Sequence[Action]|Tensor:

        states: Deque[Tensor] = deque(maxlen=4)
        q_values: List[Tensor] = []

        for observation in observations:
            states.append(self.state(observation, to_tensor=True).movedim(2,0))
            if len(states) == states.maxlen:
                q_values.append(self._policy(torch.stack(tuple(states))))

        if to_action:
            actions: List[Action] = []
            for q_values_row in q_values:
                actions.append(self._actions[int(q_values_row.argmax(dim=0).item())])
            return actions
        else:
            return torch.stack(q_values)
    
    def rollout(self, env: Asteroids, exploration_rate: float|Callable[[],float]) -> Stream[Step]:
        
        if isinstance(exploration_rate, float):
            def explore() -> bool:
                return random.random() < exploration_rate
        else:
            def explore() -> bool:
                return random.random() < exploration_rate()

        observations: Deque[Observation] = deque(maxlen=4)
        def iterator() -> Iterator[Step]:  
            while True:
                if env.running() and len(observations) == observations.maxlen:

                    if explore():
                        action: Action = random.choice(self._actions)
                    else:
                        action = self.predict(observations)[0]

                    observation,rewards = env.step(action)
                    observations.append(observation)
                    done = not env.running()

                    yield Step(observations[-2], action, rewards, observations[-1], done)

                else:
                    observation, rewards = env.reset()
                    observations.append(observation)

        return Stream(iterator())

    def train(self, 
              total_time_steps:         int,
              learning_rate:            float = 1e-4,
              learning_starts:          int = 100,
              buffer_entry_size:        int = 1_000_000,
              batch_size:               int = 32,
              tau:                      float = 1.0,
              gamma:                    float = 0.99,
              train_frequency:          Tuple[int, Literal["step", "episode"]] = (4, "step"),
              gradient_steps:           int = 1,
              target_update_frequency:  Tuple[int, Literal["step", "episode"]] = (10_000, "step"),
              exploration_rate_decay:   float = 0.1,
              initial_exploration_rate: float = 1.0,
              final_exploration_rate:   float = 0.05,
              verbose:                  bool = True) -> None:
        
        loss = torch.nn.HuberLoss()
        target_net = copy.deepcopy(self._policy)
        adam = torch.optim.Adam(target_net.parameters(), lr=learning_rate)

        exploration_rate = initial_exploration_rate

        stepper = self.rollout(Asteroids(), lambda: exploration_rate)
        
        

        for i,step in stepper.enumerate().take(total_time_steps):
            




            exploration_rate = min(final_exploration_rate, exploration_rate_decay*exploration_rate)
        
