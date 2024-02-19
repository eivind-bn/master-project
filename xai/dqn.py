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

class DQN(Agent):

    def __init__(self, device: Device, translate: bool, rotate: bool) -> None:
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
                q_values = self._policy(torch.stack(tuple(states)))

        if to_action:
            actions: List[Action] = []
            for q_values_row in q_values:
                actions.append(self._actions[int(q_values_row.argmax(dim=0).item())])
            return actions
        else:
            return torch.stack(q_values)
    
    def rollout(self, env: Asteroids) -> Stream[Step]:
        def iterator() -> Iterator[Step]:  
            if not env.running():
                observation, rewards = env.reset()
            else:
                observation = env.observation

            while True:
                action = self.predict(observation)
                next_observation,rewards = env.step(action)
                done = not env.running()
                steps.append(Step(
                    number=i,
                    observation=observation,
                    action=action,
                    rewards=rewards,
                    next_observation=next_observation,
                    done=done
                ))

                if done:
                    observation, rewards = env.reset()
                else:
                    observation = next_observation

    def train(self, 
              num_episodes: int,
              update_target_frequency: int,
              epsilon:      float,
              gamma:        float,
              batch_size:   int,
              learning_starts: float,
              buffer_entry_size:  int,
              replay_buffer_memory: Memory,
              use_ram: bool|None = None) -> None:
        env = Asteroids()

        if use_ram is None:
            use_ram = replay_buffer_memory > GigaBytes(3.0)
        
        replay_buffer: Buffer[Tuple[Observation,Action,int,Observation,bool]] = Buffer(
            entries=(),
            use_ram=use_ram,
            eviction_policy="FIFO",
            max_memory=replay_buffer_memory,
            max_entries=buffer_entry_size
        )
        huber_loss = torch.nn.SmoothL1Loss()

        learning_starts = buffer_entry_size*learning_starts
        time_step = 0

        with tqdm() as bar:
            for episode in range(num_episodes):
                time_step += 1

                if time_step % upd


                if episode % update_target_frequency == 0:
                    self._policy = policy
                    policy = copy.deepcopy(self._policy)
                    optimizer = torch.optim.Adam(policy.parameters())

                episode_reward = 0
                current_observation, rewards = env.reset()
                
                lives = env.lives()
                while env.lives() == lives:

                    if random.random() < epsilon:
                        action: Action = random.sample(population=self._actions, k=1)[0]
                    else:
                        action = self.predict(current_observation)

                    new_observation, rewards = env.step(action)
                    reward_sum = sum(reward.value for reward in rewards)
                    episode_reward += int(reward_sum)

                    small_buffer.append((current_observation, action, reward_sum, new_observation, env.running()))
                    if len(small_buffer) > 300:
                        replay_buffer.extend(entries=small_buffer, verbose=True)
                        small_buffer.clear()

                    current_observation = new_observation

                    if replay_buffer.entry_size() > learning_starts:

                        batch = replay_buffer.randoms(with_replacement=False).take(batch_size).tuple()

                        old_q = torch.stack([self.predict(entry[0], to_action=False) for entry in batch])
                        actions = torch.nn.functional.one_hot(torch.tensor([entry[1].ale_id for entry in batch], device=self._device), num_classes=len(self._actions))
                        reward_sums = torch.tensor([entry[2] for entry in batch], device=self._device).reshape((old_q.shape[0],-1))
                        done = torch.tensor([int(entry[4]) for entry in batch], device=self._device).reshape((old_q.shape[0],-1))

                        with torch.no_grad():
                            new_q = torch.stack([self.predict(entry[3], to_action=False) for entry in batch])

                        target = reward_sums + torch.mul((gamma * new_q.max(1).values.unsqueeze(1)), 1 - done)
                        current = old_q.gather(1, actions.long())

                        optimizer.zero_grad()
                        huber_loss(current, target)
                        optimizer.step()

                bar.set_description(f"{episode=}, {replay_buffer.entry_size()=}, {episode_reward=}")
                bar.update()