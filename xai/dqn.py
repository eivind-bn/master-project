from typing import *
from dataclasses import dataclass
from collections import deque
from numpy.typing import NDArray
from numpy import uint8, float32
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

class DQNStep:
    ACTIONS: Dict[Action,NDArray[float32]] = {
        Actions.NOOP:   np.array([1,0,0,0,0]),
        Actions.UP:     np.array([0,1,0,0,0]),
        Actions.LEFT:   np.array([0,0,1,0,0]),
        Actions.RIGHT:  np.array([0,0,0,1,0]),
        Actions.FIRE:   np.array([0,0,0,0,1]),
    }

    def __init__(self,    
                 observations:   Tuple[Observation,...],
                 action:         Action,
                 rewards:        Tuple[Reward,...],
                 done:           bool) -> None:
        assert len(observations) == 5, f"Incorrect number of observations: {len(observations)}"

        self.observations = observations
        self.action = action
        self.rewards = rewards
        self.done = done

    def reward_sum(self) -> int:
        return sum(reward.value for reward in self.rewards)

    def state(self, translate: bool, rotate: bool) -> NDArray[float32]:
        observations = Stream(self.observations[:-1])

        if translate:
            observations = observations.map(lambda obs: obs.translated())

        if rotate:
            observations = observations.map(lambda obs: obs.rotated())

        state = np.stack(observations.map(lambda obs: obs.numpy(normalize=True)).list())
        state = np.moveaxis(state, 3, 1).reshape((12,210,160))

        return state
    
    def next_state(self, translate: bool, rotate: bool) -> NDArray[float32]:
        observations = Stream(self.observations[1:])

        if translate:
            observations = observations.map(lambda obs: obs.translated())

        if rotate:
            observations = observations.map(lambda obs: obs.rotated())

        state = np.stack(observations.map(lambda obs: obs.numpy(normalize=True)).list())
        state = np.moveaxis(state, 3, 1).reshape((12,210,160))

        return state
    
    def numpy(self, translate: bool, rotate: bool) -> Dict[str,NDArray[np.float32]]:
        return dict(
            state=self.state(translate=translate, rotate=rotate),
            action=self.ACTIONS[self.action],
            reward=np.array([self.reward_sum()]),
            next_state = self.next_state(translate=translate, rotate=rotate),
            done=np.array([float(self.done)])
        )

class DQN(Agent):

    def __init__(self, device: Device, translate: bool, rotate: bool, replay_buffer_size: int|Memory) -> None:
        super().__init__()
        self._translate = translate
        self._rotate = rotate
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._policy = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3), stride=(1,1)), # (102, 76)
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(5,5), stride=(2,2)), # (107, 76)
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=24, out_channels=28, kernel_size=(7,7), stride=(3,3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=28, out_channels=32, kernel_size=(7,7), stride=(4,4)),
            torch.nn.ReLU(),

            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(in_features=1120, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=len(DQNStep.ACTIONS))
        ).to(device=self._device)

        self._replay_buffer = ArrayBuffer(
            capacity=replay_buffer_size,
            schema={
                "state":        ((12,210,160), "float32"),
                "action":       (len(DQNStep.ACTIONS), "float32"),
                "reward":       (1, "float32"),
                "next_state":   ((12,210,160), "float32"),
                "done":         (1, "float32")
            }
        )
    
    def rollout(self, exploration_rate: float|Callable[[],float]) -> Stream[DQNStep]:

        if isinstance(exploration_rate, float):
            def explore() -> bool:
                return random.random() < exploration_rate
        else:
            def explore() -> bool:
                return random.random() < exploration_rate()
        
        def iterator() -> Iterator[DQNStep]:
            env = Asteroids()
            observations: Deque[Observation] = deque(maxlen=5)
            actions = tuple(DQNStep.ACTIONS)

            action: Action = Actions.NOOP
            rewards: Tuple[Reward,...]= tuple()
            running = env.running()

            while len(observations) < observations.maxlen:
                if running:
                    action = random.choice(actions)
                    observation,rewards = env.step(action)
                else:
                    observation, rewards = env.reset()
                    running = env.running()

                observations.append(observation)

            step = DQNStep(
                observations=tuple(observations),
                action=action,
                rewards=rewards,
                done=not running
            )

            yield step

            while True:
                if running:
                    if explore():
                        action = random.choice(actions)
                    else:
                        state = step.next_state(translate=self._translate, rotate=self._rotate)
                        tensor_state = torch.from_numpy(state).to(
                            dtype=torch.float32,
                            device=self._device
                        ).unsqueeze(0)

                        policy: Tensor = self._policy(tensor_state)

                        action = actions[int(policy.argmax(1).item())]

                    observation,rewards = env.step(action)

                else:
                    observation, rewards = env.reset()
                    running = env.running()

                observations.append(observation)

                step = DQNStep(
                    observations=tuple(observations),
                    action=action,
                    rewards=rewards,
                    done=not running
                )

                yield step

        return Stream(iterator())

    def train(self, 
              total_time_steps:         int,
              learning_rate:            float = 1e-4,
              learning_starts:          int = 100,
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
        
        exploration_rate = initial_exploration_rate
        stepper = self.rollout(lambda: exploration_rate)
        
        target_policy = copy.deepcopy(self._policy)
        adam = torch.optim.Adam(target_policy.parameters(), lr=learning_rate)
        huber_loss = torch.nn.HuberLoss()

        match train_frequency:
            case (steps, "step"):
                def train(step: int, episode: int) -> bool:
                    return step % train_frequency[0] == 0
            case (episodes, "episode"):
                def train(step: int, episode: int) -> bool:
                    return episode % train_frequency[0] == 0
            case _:
                assert_never(train_frequency)

        match target_update_frequency:
            case (steps, "step"):
                def target_update(step: int, episode: int) -> bool:
                    return step % target_update_frequency[0] == 0
            case (episode, "episode"):
                def target_update(step: int, episode: int) -> bool:
                    return episode % target_update_frequency[0] == 0
            case _:
                assert_never(target_update_frequency)


        initial_steps = stepper.take(learning_starts)\
            .flatmap(lambda step: step.numpy(translate=self._translate, rotate=self._rotate).items())\
            .group_by(keep_key=True)\
            .map(lambda kvi: (kvi[0], np.stack(kvi[1])))

        self._replay_buffer.append(dict(initial_steps))
        total_reward = 0
        episode = 0
        for i,step in stepper.enumerate().take(total_time_steps):
            if step.done:
                print(f"{episode=}, {total_reward=}")
                total_reward = 0
                episode += 1

            total_reward += step.reward_sum()

            self._replay_buffer.append(step.numpy(translate=self._translate, rotate=self._rotate))

            if train(i, episode):
                for epoch in range(gradient_steps):
                    adam.zero_grad()

                    batch = self._replay_buffer.mini_batch(batch_size)
                    state = torch.from_numpy(batch["state"]).to(dtype=torch.float32, device=self._device)
                    action = torch.from_numpy(batch["action"]).to(dtype=torch.float32, device=self._device).bool()
                    reward = torch.from_numpy(batch["reward"]).to(dtype=torch.float32, device=self._device)
                    next_state = torch.from_numpy(batch["next_state"]).to(dtype=torch.float32, device=self._device)
                    done = torch.from_numpy(batch["done"]).to(dtype=torch.float32, device=self._device)

                    Q: Tensor = self._policy(state)

                    with torch.no_grad():
                        Q_next: Tensor = target_policy(next_state)
                        Q_next, _ = Q_next.max(dim=1)
                        Q_next = Q_next.reshape((-1,1))
                        Q_target = reward + (1 - done) * gamma * Q_next

                    loss: Tensor = huber_loss(Q[action], Q_target.flatten())
                    loss.backward()
                    adam.step()

            if target_update(i, episode):
                self._policy = target_policy
                target_policy = copy.deepcopy(target_policy)
                adam = torch.optim.Adam(target_policy.parameters(), lr=learning_rate)
