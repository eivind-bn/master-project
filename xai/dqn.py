from typing import *
from collections import deque
from numpy.typing import NDArray
from numpy import uint8
from torch import Tensor
from .policy import Policy
from .action import Actions, Action
from .asteroids import Asteroids
from .window import Window, WindowClosed
from .record import Recorder
from .observation import Observation
from .explainer import Explainer
from .reward import Reward
from .feed_forward import FeedForward

import torch
import random
import numpy as np

class DQN:

    def __init__(self, translate: bool, rotate: bool) -> None:
        self._translate = translate
        self._rotate = rotate
        self._actions = (
            Actions.NOOP,
            Actions.UP,
            Actions.LEFT,
            Actions.RIGHT,
            Actions.FIRE,
        )
        self._policy = Policy.new(
            input_dim=Asteroids.observation_shape,
            output_dim=len(self._actions),
            hidden_layers=[2**8,2**6,2**4],
            normalize=255.0
            )
        
    def Q(self, observation: Observation) -> Tensor:
        if self._translate:
            observation = observation.translated()

        if self._rotate:
            observation = observation.rotated()

        return self._policy.predict(observation.numpy()).tensor()

    def predict(self, observation: Observation) -> Action:
        return self._actions[int(self.Q(observation).argmax(dim=0))]

    def train(self, 
              buffer_size:  int,
              num_episodes: int,
              epsilon:      float,
              gamma:        float,
              steps:        int,
              batch_size:   int,
              alpha:        float) -> None:
        env = Asteroids()
        replay_buffer: Deque[Tuple[Observation,Action,int,Observation]] = deque(maxlen=buffer_size)
        adam = self._policy.adam()
        
        for episode in range(num_episodes):
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
                replay_buffer.append((current_observation, action, reward_sum, new_observation))
                current_observation = new_observation

                if len(replay_buffer) > batch_size:
                    states = np.stack([entry[0].numpy() for entry in replay_buffer])
                    old_q = torch.stack([self.Q(entry[0]) for entry in replay_buffer])
                    new_q = torch.stack([self.Q(entry[3]) for entry in replay_buffer])
                    reward_sums = torch.tensor([entry[2] for entry in replay_buffer]).reshape((states.shape[0],-1))
                    actions = torch.zeros(size=(states.shape[0],len(self._actions)), dtype=torch.int, device=self._policy.device)
                    actions[torch.arange(0,states.shape[0]),action.ale_id] = 1.0

                    target_q = old_q.clone()
                    target_q[:,actions] = old_q[:,actions] + alpha*(reward_sums + gamma*new_q.max(dim=1)[0] - old_q[:,actions])

                    adam.fit(
                        X=states,
                        Y=target_q,
                        batch_size=batch_size,
                        steps=steps,
                        loss_criterion="HuberLoss",
                    )
                    replay_buffer.clear()

            print(f"Episode {episode}: {episode_reward}")

    def play(self, 
             record_path:   str|None = None, 
             show:          bool = False, 
             rollout:       bool = False) -> None:
        with Window(name=self.__class__.__name__, fps=60, scale=4.0) as window:
            with Recorder(filename=record_path) as recorder:
                try:
                    while True:
                        env = Asteroids()
                        observation, rewards = env.reset()
                        while env.running():
                            window.update(observation.numpy())
                            action = self.predict(observation)
                            observation, rewards = env.step(action)
                except WindowClosed:
                    pass

    def save(self, path: str) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> Self:
        dqn: Self = torch.load(path)
        if isinstance(dqn, cls):
            return dqn
        else:
            raise TypeError(f"Invalid type: {type(dqn)}")
        