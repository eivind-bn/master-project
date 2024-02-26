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
    def __init__(self,    
                 observation:   Observation,
                 action:        Action,
                 rewards:       Tuple[Reward,...],
                 done:          bool,
                 state:         Tensor,
                 next_state:    Tensor,
                 action_id:     int) -> None:
        self.observation = observation
        self.action = action
        self.rewards = rewards
        self.done = done
        self.state = state
        self.next_state = next_state
        self.action_id = action_id

    def reward_sum(self) -> int:
        return sum(reward.value for reward in self.rewards)
    
    def numpy(self) -> Dict[str,NDArray]:
        return {
            "state": self.state.numpy(force=True),
            "action": np.array([self.action_id]),
            "reward": np.array([self.reward_sum()]),
            "next_state": self.next_state.numpy(force=True),
            "done": np.array([self.done])
        }

class DQN(Agent):

    def __init__(self, autoencoder_path: str, device: Device, translate: bool, rotate: bool) -> None:
        super().__init__()
        self._translate = translate
        self._rotate = rotate
        self._actions = (
                Actions.NOOP,
                Actions.UP,
                Actions.LEFT,
                Actions.RIGHT,
                Actions.FIRE
                )
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._autoencoder: torch.nn.Sequential = torch.load(autoencoder_path, map_location=self._device)

        self._encoder = self._autoencoder[:7]
        self._decoder = self._autoencoder[7:]

        self._policy = torch.nn.Sequential(
            torch.nn.Linear(in_features=4*32, out_features=2*32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2*32, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=len(self._actions))
        ).to(device=self._device)
    
    def rollout(self, 
                exploration_rate: float|Callable[[],float], 
                frame_skips: int) -> Stream[DQNStep]:

        if isinstance(exploration_rate, float):
            def explore() -> bool:
                return random.random() < exploration_rate
        else:
            def explore() -> bool:
                return random.random() < exploration_rate()
        
        def iterator() -> Iterator[DQNStep]:
            env = Asteroids()
            observation, rewards = env.reset()
            latents: Deque[Tensor] = deque(maxlen=4)
            
            def encode(observation: Observation) -> Tensor:
                if self._translate:
                    observation = observation.translated()
                if self._rotate:
                    observation = observation.rotated()

                embedding: Tensor = self._encoder(observation.tensor(normalize=True, device=self._device).unsqueeze(0))
                embedding = embedding.squeeze(0)

                return embedding

            while len(latents) < latents.maxlen:
                if env.running():
                    action_id = random.randrange(0, len(self._actions))
                    action = self._actions[action_id]
                    observation,rewards = env.step(action)
                    latents.append(encode(observation))
                else:
                    env.reset()

            state = torch.stack(tuple(latents)).reshape((1,-1))

            skip_cnt = 0
            action_id = random.randrange(0, len(self._actions))
            action = self._actions[action_id]
            while True:
                if env.running():
                    if skip_cnt < frame_skips:
                        skip_cnt += 1
                        action_id = random.randrange(0, len(self._actions))
                        action = self._actions[action_id]
                    else:
                        skip_cnt = 0
                        if explore():
                            action_id = random.randrange(0, len(self._actions))
                            action = self._actions[action_id]
                        else:
                            q_values: Tensor = self._policy(state)
                            action_id = int(q_values.argmax(1).item())
                            action = self._actions[action_id]

                    observation,rewards = env.step(action)
                    latents.append(encode(observation))

                    next_state = torch.stack(tuple(latents)).reshape((1,-1))

                    yield DQNStep(
                        observation=observation,
                        action=action,
                        rewards=rewards,
                        done=not env.running(),
                        state=state,
                        next_state=next_state,
                        action_id=action_id
                    )
                else:
                    env.reset()

        return Stream(iterator())

    def train(self, 
              total_time_steps:                 int,
              replay_buffer_size:               int|Memory,
              learning_rate:                    float = 1e-4,
              learning_starts:                  int = 100,
              batch_size:                       int = 32,
              tau:                              float = 1.0,
              gamma:                            float = 0.99,
              frame_skip:                       int = 4,
              save_frequency:                   Tuple[int, Literal["step", "episode"], str]|None = None,
              train_frequency:                  Tuple[int, Literal["step", "episode"]] = (4, "step"),
              gradient_steps:                   int = 1,
              target_update_frequency:          Tuple[int, Literal["step", "episode"]] = (10_000, "step"),
              initial_exploration_rate:         float = 1.0,
              final_exploration_rate:           float = 0.05,
              final_exploration_rate_progress:  float = 0.75,
              verbose:                          bool = True) -> None:
        
        exploration_rate = initial_exploration_rate
        
        er0 = initial_exploration_rate
        er = final_exploration_rate
        ts = final_exploration_rate_progress*total_time_steps

        stepper = self.rollout(lambda: exploration_rate, frame_skips=frame_skip)
        
        target_policy = copy.deepcopy(self._policy)
        online_network = copy.deepcopy(self._policy)
        adam = torch.optim.Adam(online_network.parameters(), lr=learning_rate)
        huber_loss = torch.nn.HuberLoss()

        replay_buffer = ArrayBuffer(
            capacity=replay_buffer_size,
            schema={
                "state":        ((4*32), "float32"),
                "action":       (1, "uint8"),
                "reward":       (1, "float32"),
                "next_state":   ((4*32), "float32"),
                "done":         (1, "float32")
            }
        )

        match save_frequency:
            case (steps, "step", path):
                def save(step: int, episode: int) -> None:
                    if step % save_frequency[0] == 0:
                        self.save(save_frequency[2])
            case (episodes, "episode", path):
                def save(step: int, episode: int) -> None:
                    if episode % save_frequency[0] == 0:
                        self.save(save_frequency[2])
            case _:
                def save(step: int, episode: int) -> None:
                    pass
        
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

        with tqdm(total=learning_starts, desc="Filling replay buffer: ", disable=not verbose) as bar:
            for step in stepper.take(learning_starts):
                replay_buffer.append(step.numpy())
                
                bar.update()

        total_reward = 0
        episode = 0
        with tqdm(total=total_time_steps, disable=not verbose) as bar:
            for i,step in stepper.enumerate().take(total_time_steps):
                if step.done:
                    total_reward = 0
                    episode += 1

                total_reward += step.reward_sum()

                replay_buffer.append(step.numpy())

                if train(i, episode):
                    for epoch in range(gradient_steps):
                        adam.zero_grad()

                        batch = replay_buffer.mini_batch(batch_size)
                        state = torch.from_numpy(batch["state"]).to(dtype=torch.float32, device=self._device)
                        action = torch.from_numpy(batch["action"]).to(dtype=torch.float32, device=self._device).long()
                        reward = torch.from_numpy(batch["reward"]).to(dtype=torch.float32, device=self._device)
                        next_state = torch.from_numpy(batch["next_state"]).to(dtype=torch.float32, device=self._device)
                        done = torch.from_numpy(batch["done"]).to(dtype=torch.float32, device=self._device)

                        Q: Tensor = online_network(state)
                        Q = Q.gather(dim=1, index=action)

                        with torch.no_grad():
                            Q_next: Tensor = target_policy(next_state)
                            Q_next, _ = Q_next.max(dim=1)
                            Q_next = Q_next.reshape((-1,1))
                            Q_target = reward + (1 - done) * gamma * Q_next
                            
                        loss: Tensor = huber_loss(Q, Q_target)
                        loss.backward()
                        adam.step()

                if target_update(i, episode):
                    self._policy = online_network
                    target_policy = copy.deepcopy(online_network)
                    online_network = copy.deepcopy(online_network)
                    adam = torch.optim.Adam(online_network.parameters(), lr=learning_rate)

                save(i, episode)

                exploration_rate = max(er0 + ((er - er0)/(ts)*i), 0.05)
                bar.update()
                bar.set_description(f"time_step={i}, {episode=}, {total_reward=}, exploration={exploration_rate:.2f}")