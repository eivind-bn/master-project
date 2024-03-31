from . import *
from collections import deque
from numpy.typing import NDArray
from dataclasses import dataclass
from torch import Tensor
from tqdm import tqdm

import torch
import random
import numpy as np
import copy

class DQN(Agent):
    @dataclass
    class Step(Agent.Step):
        state:      Tensor
        next_state: Tensor
        action_id:  int

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

        X: TypeAlias = Tuple[Literal[210], Literal[160], Literal[3]]
        L: TypeAlias = Tuple[Literal[32]]

        self._autoencoder: AutoEncoder[X,L] = AutoEncoder.load(autoencoder_path)
        self._policy = Network.dense(input_dim=(4*32,), output_dim=(5,), device=device)
    
    def rollout(self, 
                exploration_rate: float|Callable[[],float], 
                frame_skips: int) -> Stream["DQN.Step"]:

        if isinstance(exploration_rate, float):
            def explore() -> bool:
                return random.random() < exploration_rate
        else:
            def explore() -> bool:
                return random.random() < exploration_rate()
        
        def iterator() -> Iterator[DQN.Step]:
            env = Asteroids()
            observation, rewards = env.reset()
            latents: Deque[Tensor] = deque(maxlen=4)
            assert latents.maxlen is not None
            
            def encode(observation: Observation) -> Tensor:
                if self._translate:
                    observation = observation.translated()
                if self._rotate:
                    observation = observation.rotated()

                embedding: Tensor = self._autoencoder.encoder(observation.tensor(normalize=True, device=self._device).unsqueeze(0)).output()
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
                            q_values: Tensor = self._policy(state).output()
                            action_id = int(q_values.argmax(1).item())
                            action = self._actions[action_id]

                    observation,rewards = env.step(action)
                    latents.append(encode(observation))

                    next_state = torch.stack(tuple(latents)).reshape((1,-1))

                    yield DQN.Step(
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
              learning_starts:                  int = 3_000,
              batch_size:                       int = 32,
              tau:                              float = 1.0,
              gamma:                            float = 0.99,
              frame_skip:                       int = 4,
              episode_save_freq:                int = 5,
              save_path:                        str|None = None,
              train_frequency:                  int = 32,
              gradient_steps:                   int = 1,
              target_update_frequency:          int = 2000,
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

        with tqdm(total=learning_starts, desc="Filling replay buffer: ", disable=not verbose) as bar:
            for step in stepper.take(learning_starts):
                replay_buffer.append(step.numpy())
                bar.update()

        total_reward = 0
        episode = 0

        with tqdm(total=total_time_steps, disable=not verbose) as bar:
            for time_step,step in stepper.enumerate().take(total_time_steps):

                if step.done:
                    total_reward = 0
                    episode += 1
                    if save_path is not None and episode % episode_save_freq == 0:
                        self.save(save_path)

                total_reward += step.reward_sum()

                replay_buffer.append(step.numpy())

                if time_step % train_frequency == 0:
                    for epoch in range(gradient_steps):
                        adam.zero_grad()

                        batch = replay_buffer.mini_batch(batch_size)
                        state = torch.from_numpy(batch["state"]).to(dtype=torch.float32, device=self._device)
                        action = torch.from_numpy(batch["action"]).to(dtype=torch.float32, device=self._device).long()
                        reward = torch.from_numpy(batch["reward"]).to(dtype=torch.float32, device=self._device)
                        next_state = torch.from_numpy(batch["next_state"]).to(dtype=torch.float32, device=self._device)
                        done = torch.from_numpy(batch["done"]).to(dtype=torch.float32, device=self._device)

                        Q: Tensor = online_network(state).output()
                        Q = Q.gather(dim=1, index=action)

                        with torch.no_grad():
                            Q_next: Tensor = target_policy(next_state).output()
                            Q_next, _ = Q_next.max(dim=1)
                            Q_next = Q_next.reshape((-1,1))
                            Q_target = reward + (1 - done) * gamma * Q_next
                            
                        loss: Tensor = huber_loss(Q, Q_target)
                        loss.backward()
                        adam.step()

                if time_step % target_update_frequency == 0:
                    self._policy = online_network
                    target_policy = copy.deepcopy(online_network)
                    online_network = copy.deepcopy(online_network)
                    adam = torch.optim.Adam(online_network.parameters(), lr=learning_rate)

                exploration_rate = max(er0 + ((er - er0)/(ts)*time_step), 0.05)
                bar.update()
                bar.set_description(f"time_step={time_step}, {episode=}, {total_reward=}, exploration={exploration_rate:.2f}")