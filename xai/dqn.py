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

Sx = Tuple[Literal[210], Literal[160], Literal[3]]
Sy = Tuple[Literal[5]]
Sz = Tuple[Literal[32]]

class Explanations(NamedTuple):
    latent_explanation:     Explanation[Sz,Sx]
    policy_explanation:     Explanation[Sy,Sz]
    eap_explanation:        Explanation[Sy,Sx]
    ecp_explanation:        Explanation[Sy,Sx]


class DQN(Agent[Sx,Sy]):
    @dataclass
    class Step(Agent.Step):
        parent:     "DQN"
        state:      Tensor
        next_state: Tensor
        action_id:  int

        def explain_latents(self, algorithm: Explainer|Explainers, background: Array|None = None) -> Explanation:
            return self.parent._autoencoder.decoder(self.next_state[-1]).explain(algorithm, background)
        
        def explain(self, 
                    algorithm:          Explainer|Explainers, 
                    decoder_background: Array|None = None,
                    q_background:       Array|None = None) -> Explanations:
            l_explanation = self.explain_latents(algorithm, decoder_background)
            q_explanation = self.parent._policy(self.next_state).explain(algorithm, q_background)

            q_shap_values = q_explanation.flatten().shap_values.reshape((5,4,32)).sum(1)
            attribution_weights = l_explanation.flatten().attribution_weights().shap_values.T
            contribution_weights = l_explanation.flatten().contribution_weights().shap_values.T
            return Explanations(
                latent_explanation=l_explanation,
                policy_explanation=q_explanation,
                eap_explanation=Explanation(
                    class_shape=(5,),
                    feature_shape=(210,160),
                    shap_values=(q_shap_values@attribution_weights).reshape((5,210,160,3)).sum(3),
                    compute_time=l_explanation.compute_time + q_explanation.compute_time
                    ),
                ecp_explanation=Explanation(
                    class_shape=(5,),
                    feature_shape=(210,160),
                    shap_values=(q_shap_values@contribution_weights).reshape((5,210,160,3)).sum(3),
                    compute_time=l_explanation.compute_time + q_explanation.compute_time
                )
            )

        def reward_sum(self) -> int:
            return sum(reward.value for reward in self.rewards)
        
        def count_none_zero_rewards(self) -> int:
            return sum(1 for reward in self.rewards if reward != Reward.NONE)
        
        def numpy(self) -> Dict[str,NDArray]:
            return {
                "state": self.state.numpy(force=True),
                "action": np.array([self.action_id]),
                "reward": np.array([self.reward_sum()]),
                "next_state": self.next_state.numpy(force=True),
                "done": np.array([self.done or self.observation.spaceship_crashed]),
            }

    def __init__(self, autoencoder_path: str, device: Device, translate: bool, rotate: bool) -> None:
        
        autoencoder: AutoEncoder[Sx,Literal[32]] = AutoEncoder.load(autoencoder_path, device=device)
        policy = Network.dense(input_dim=(4,32), output_dim=(5,), device=device)

        super().__init__(
            device=device,
            input_shape=(4,) + autoencoder.input_shape,
            output_shape=policy.output_shape,
            logits=Lambda(
                f=lambda tensor: policy(autoencoder(tensor)).output(),
                repr=lambda _: f"Stacking()"
                ),
            )
        
        self._translate = translate
        self._rotate = rotate
        self._actions = (
                Actions.NOOP,
                Actions.UP,
                Actions.LEFT,
                Actions.RIGHT,
                Actions.FIRE
                )
        self.rewards_per_episode: List[int] = []
        self.exploration_rate_per_episode: List[float] = []

        self._autoencoder = autoencoder
        self._policy = policy
    
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

                embedding: Tensor = self._autoencoder.encoder(observation.tensor(normalize=True, device=self.device).unsqueeze(0)).output()
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

            state = torch.stack(tuple(latents)).reshape((4,32))

            skip_cnt = 0
            action_id = random.randrange(0, len(self._actions))
            action = self._actions[action_id]
            while True:
                if env.running():
                    if skip_cnt < frame_skips:
                        skip_cnt += 1
                    else:
                        skip_cnt = 0
                        if explore():
                            action_id = random.randrange(0, len(self._actions))
                            action = self._actions[action_id]
                        else:
                            q_values = self._policy(state)
                            action_id = int(q_values().argmax(0).item())
                            action = self._actions[action_id]

                    observation,rewards = env.step(action)
                    latents.append(encode(observation))

                    next_state = torch.stack(tuple(latents)).reshape((4,32))

                    yield DQN.Step(
                        parent=self,
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
              max_episodes:                     int|None = None,
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
              verbose:                          bool = True,
              q_value_head_background_freq:     int = 25,
              q_value_head_background_path:     str|None = None) -> None: # type: ignore
        
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
                "state":        ((4,32), "float32"),
                "action":       (1, "uint8"),
                "reward":       (1, "float32"),
                "next_state":   ((4,32), "float32"),
                "done":         (1, "float32"),
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
                    self.rewards_per_episode.append(total_reward)
                    self.exploration_rate_per_episode.append(exploration_rate)
                    total_reward = 0
                    if episode % q_value_head_background_freq == 0 and q_value_head_background_path:
                        states = replay_buffer.data(False)["state"]
                        np.save(q_value_head_background_path, states)

                    episode += 1
                    if save_path is not None and episode % episode_save_freq == 0:
                        self.save(save_path)

                    if max_episodes is not None and episode >= max_episodes:
                        self.save(save_path)
                        return

                total_reward += step.reward_sum()

                replay_buffer.append(step.numpy())

                if time_step % train_frequency == 0:
                    for epoch in range(gradient_steps):
                        adam.zero_grad()

                        batch = replay_buffer.mini_batch(batch_size)
                        state = torch.from_numpy(batch["state"]).to(dtype=torch.float32, device=self.device)
                        action = torch.from_numpy(batch["action"]).to(dtype=torch.float32, device=self.device).long()
                        reward = torch.from_numpy(batch["reward"]).to(dtype=torch.float32, device=self.device)
                        next_state = torch.from_numpy(batch["next_state"]).to(dtype=torch.float32, device=self.device)
                        done = torch.from_numpy(batch["done"]).to(dtype=torch.float32, device=self.device)

                        Q: Tensor = online_network(state).output()
                        Q = Q.gather(dim=1, index=action)
                        
                        reward = torch.where(reward > 0, 1, reward)
                        reward = torch.where(done > 0, -1, reward)

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