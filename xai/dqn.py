from typing import *
from collections import deque
from numpy.typing import NDArray
from numpy import uint8
from torch import Tensor
from tqdm import tqdm
from .policy import Policy
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

import torch
import random
import numpy as np

class DQN(Agent):

    def __init__(self, translate: bool, rotate: bool) -> None:
        super().__init__()
        self._translate = translate
        self._rotate = rotate
        self._actions = (
            Actions.NOOP,
            Actions.UP,
            Actions.LEFT,
            Actions.RIGHT,
            Actions.FIRE,
        )
        hidden_layers = [2**8,2**6,2**4]
        self._encoder = Policy.new(
            input_dim=Asteroids.observation_shape,
            output_dim=10,
            hidden_layers=hidden_layers
            )
        self._decoder = Policy.new(
            input_dim=self._encoder.output_dim,
            output_dim=Asteroids.observation_shape,
            hidden_layers=hidden_layers[::-1],
            )
        self._policy = Policy.new(
            input_dim=self._encoder.output_dim,
            output_dim=len(self._actions)
            )
        self.device = self._policy.device
        
    def state(self, observation: Observation) -> NDArray[np.float32]:
        if self._translate:
            observation = observation.translated()

        if self._rotate:
            observation = observation.rotated()

        return observation.numpy(normalize=True)
        
    def Q(self, observation: Observation) -> Tensor:
        return (self._encoder + self._policy).predict(self.state(observation)).tensor()

    def predict(self, observation: Observation) -> Action:
        return self._actions[int(self.Q(observation).argmax(dim=0))]
    
    @overload
    def rollout(self,
                env: Asteroids, 
                time_steps: int) -> Tuple[Step,...]: ...
        
    @overload
    def rollout(self,
                env: Asteroids, 
                time_steps: int,
                deque: Deque[Step]) -> None: ...
    
    def rollout(self,
                env: Asteroids, 
                time_steps: int,
                deque: Deque[Step]|None = None) -> Tuple[Step,...]|None:
        
        if deque is None:
            steps: List[Step]|Deque[Step] = []
        else:
            steps = deque
        
        if not env.running():
            observation, rewards = env.reset()
        else:
            observation = env.observation
        
        for i in range(time_steps):
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

        if deque is None:
            return tuple(steps)
        else:
            return None


    def train_autoencoder(self, 
                          buffer_size:  int, 
                          time_steps:   int, 
                          episodes:     int,
                          epochs:       int,
                          batch_size:   int,
                          **params:     Unpack[Adam.Params]) -> None:
        encoder, decoder = self._encoder.copy(), self._decoder.copy()
        adam = (encoder + decoder).adam(**params)
        env = Asteroids()
        replay_buffer: Deque[Step] = deque(maxlen=buffer_size)
        for episode in range(episodes):
            self.rollout(env, time_steps, replay_buffer)
            states = np.stack([self.state(step.observation) for step in replay_buffer])
            adam.fit(
                X=states,
                Y=states,
                epochs=epochs,
                batch_size=batch_size,
                loss_criterion="MSELoss",
                verbose=True
            )
        self._encoder, self._decoder = encoder, decoder


    def train(self, 
              buffer_size:  int,
              num_episodes: int,
              epsilon:      float,
              gamma:        float,
              steps:        int,
              batch_size:   int,
              sample_prob:  float,
              learning_starts: float,
              samples_per_train: float,
              alpha:        float) -> None:
        env = Asteroids()
        replay_buffer: Deque[Tuple[Observation,Action,int,Observation]] = deque(maxlen=buffer_size)
        policy = self._policy
        encoder = self._encoder
        decoder = self._decoder
        sample_count = 0
        learning_starts = buffer_size*learning_starts
        
        with tqdm() as bar:
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

                    if random.random() < sample_prob:
                        replay_buffer.append((current_observation, action, reward_sum, new_observation))
                        sample_count += 1

                    current_observation = new_observation

                    if len(replay_buffer) > learning_starts and sample_count > samples_per_train:
                        sample_count  = 0
                        states = np.stack([self.state(entry[0]) for entry in replay_buffer])
                        # old_q = torch.stack([self.Q(entry[0]) for entry in replay_buffer])
                        # actions = torch.tensor([entry[1].ale_id for entry in replay_buffer], device=policy.device)
                        # new_q = torch.stack([self.Q(entry[3]) for entry in replay_buffer])
                        # reward_sums = torch.tensor([entry[2] for entry in replay_buffer], device=policy.device).reshape((states.shape[0],-1))

                        # target_q = old_q.clone()
                        # target_q[:,actions] = old_q[:,actions] + alpha*(reward_sums + gamma*new_q.max(dim=1)[0] - old_q[:,actions])

                        (encoder + decoder).adam().fit(
                            X=states,
                            Y=states,
                            batch_size=batch_size,
                            epochs=steps,
                            loss_criterion="MSELoss",
                            verbose=True
                        )

                        # policy.adam().fit(
                        #     X=encoder.predict(states),
                        #     Y=target_q,
                        #     batch_size=batch_size,
                        #     steps=steps,
                        #     loss_criterion="HuberLoss",
                        # )

                bar.set_description(f"{episode=}, {len(replay_buffer)=}")
                bar.update()

    def play(self, 
             record_path:   str|None = None, 
             show_decode:   bool = False,
             show:          bool = False, 
             rollout:       bool = False) -> None:
        with Window(name=self.__class__.__name__, fps=60, scale=4.0) as window:
            with Recorder(filename=record_path) as recorder:
                try:
                    autoencoder = self._encoder + self._decoder
                    while True:
                        env = Asteroids()
                        observation, rewards = env.reset()
                        while env.running():
                            if show_decode:
                                state = self.state(observation)
                                recon = autoencoder.predict(state).numpy()
                                min, max = recon.min(), recon.max()
                                recon = ((recon - min)/(max - min))*255
                                recon = recon.astype(np.uint8)
                                window.update(np.hstack([state,recon]))
                            else:
                                window.update(observation.numpy(normalize=False))
                            action = self.predict(observation)
                            observation, rewards = env.step(action)
                except WindowClosed:
                    pass