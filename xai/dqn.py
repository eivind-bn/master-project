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
from .buffer import Buffer
from .bytes import GigaBytes, Memory

import torch
import random
import numpy as np
import copy

class ReplayBatch(NamedTuple):
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    done: Tensor

class ReplayBuffer:

    def __init__(self, size: int, state_shape: Tuple[int,...], actions: Sequence[Action]) -> None:
        self.state = torch.zeros((size,) + state_shape, dtype=torch.float32, device="cpu")
        self.action = torch.zeros((size, len(actions)), dtype=torch.float32, device="cpu")
        self.reward = torch.zeros((size, 1), dtype=torch.float32, device="cpu")
        self.next_state = torch.zeros((size,) + state_shape, dtype=torch.float32, device="cpu")
        self.done = torch.zeros((size, 1))

        self._action_to_idx = {action:index for index,action in enumerate(actions)}
        self._size = size
        self._current_position = 0

    @overload
    def append(self, 
               *,
               steps:        Iterable[Step]) -> None: ...

    @overload
    def append(self, 
               *,
               state:       Tensor, 
               action:      Tensor, 
               reward:      Tensor, 
               next_state:  Tensor, 
               done:        Tensor) -> None: ...

    def append(self, 
               *,
               steps:       Iterable[Step]|None = None, 
               state:       Tensor|None = None, 
               action:      Tensor|None = None, 
               reward:      Tensor|None = None, 
               next_state:  Tensor|None = None, 
               done:        Tensor|None = None) -> None:
        if steps is None:
            assert all(arg is not None for arg in (state, action, reward, next_state, done))
            assert len(state) == len(action) == len(reward) == len(next_state) == len(done)

            indices = torch.arange(self._current_position, self._current_position + len(state), dtype=torch.int32, device="cpu") % self._size

            self.state[indices] = state.to(dtype=torch.float32, device="cpu")
            self.action[indices] = action.to(dtype=torch.float32, device="cpu")
            self.reward[indices] = reward.to(dtype=torch.float32, device="cpu")
            self.next_state[indices] = next_state.to(dtype=torch.float32, device="cpu")
            self.done[indices] = done.to(dtype=torch.float32, device="cpu")

            if len(indices) > 0:
                self._current_position = int(indices[-1].item())
        else:
            self.append(
                state=torch.stack([step.observation.translated().rotated().tensor(normalize=True, device="cpu") for step in steps]),
                action=torch.tensor([self._action_to_idx[step.action] for step in steps], dtype=torch.int32, device="cpu"),
                reward=torch.tensor([step.reward_sum() for step in steps], dtype=torch.float32, device="cpu"),
                next_state=torch.stack([step.next_observation.translated().rotated().tensor(normalize=True, device="cpu") for step in steps]),
                done=torch.tensor([float(step.done) for step in steps], dtype=torch.float32, device="cpu"),
            )

    def batch(self, size: int, device: Device) -> ReplayBatch:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        selections = torch.randperm(len(self.state), device="cpu")[:size]

        return ReplayBatch(
            state=self.state[selections].to(device=device),
            action=self.action[selections].to(device=device),
            reward=self.reward[selections].to(device=device),
            next_state=self.next_state[selections].to(device=device),
            done=self.done[selections].to(device=device)
        )


class DQN(Agent):

    def __init__(self, device: Device, translate: bool, rotate: bool, replay_buffer_size: int) -> None:
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

        self._replay_buffer = ReplayBuffer(
            size=replay_buffer_size,
            state_shape=(4,210,160,3),
            actions=self._actions
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
        
        self._replay_buffer.append(steps=stepper.take(learning_starts))

        for i,step in stepper.enumerate().take(total_time_steps):
            




            exploration_rate = min(final_exploration_rate, exploration_rate_decay*exploration_rate)
        
