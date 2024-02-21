from typing import *
from torch import Tensor

from xai.bytes import Memory, GigaBytes
from xai.cache import Cache
from xai.policy import Device
from .buffer import Buffer
from .step import Step
from .observation import Observation
from .action import Action
from .reward import Reward

import torch

ReplayBatch: TypeAlias = Tuple[
    Tuple[Observation,...], 
    Tuple[Action,...], 
    Tuple[Tuple[Reward,...]], 
    Tuple[Observation,...], 
    Tuple[bool,...]
    ]

ReplayTensorBatch: TypeAlias = Tuple[
    Tensor, 
    Tensor, 
    Tensor, 
    Tensor, 
    Tensor
    ]

class ReplayBuffer(Buffer[Step]):
    
    def __init__(self, 
                 entries:           Iterable[Step],
                 action_mapping:    Sequence[Action],
                 use_ram:           bool, 
                 max_memory:        Memory = GigaBytes(5.0), 
                 max_entries:       int = 10_000) -> None:
        super().__init__(
            entries=entries, 
            eviction_policy="FIFO", 
            use_ram=use_ram, 
            max_memory=max_memory, 
            max_entries=max_entries)
        
        self.action_mapping = {action:index for index,action in enumerate(action_mapping)}
        
    def record(self, 
               observation: Observation, 
               action: Action, 
               rewards: Sequence[Reward], 
               next_observation: Observation, 
               done: bool) -> None:
        self.append(
            entry=Step(
                observation=observation,
                action=action,
                rewards=tuple(rewards),
                next_observation=next_observation,
                done=done
            )
        )

    @overload
    def replay_batch(self, 
                     size: int, 
                     to_tensor: Literal[False]) -> ReplayBatch: ...

    @overload
    def replay_batch(self, 
                     size: int, 
                     to_tensor: Literal[True], 
                     device: Device, 
                     normalize: bool = ...) -> ReplayTensorBatch: ...

    def replay_batch(self, 
                     size: int, 
                     to_tensor: bool,
                     device: Device|None = None, 
                     normalize: bool = True) -> ReplayBatch|ReplayTensorBatch:
        if to_tensor:
            assert device is not None

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            observations, actions, rewards, next_observations, dones = self.replay_batch(size=size, to_tensor=False)

            action_idx = self.action_mapping
            action_cnt = len(self.action_mapping)

            obs_tensor = torch.stack([obs.tensor(normalize=normalize, device="cpu") for obs in observations]).to(device=device)
            actions_tensor = torch.tensor([action_idx[action] for action in actions]).to(device=device)
            actions_tensor = torch.nn.functional.one_hot(actions_tensor, num_classes=action_cnt)
            rewards_tensor = torch.tensor([sum(rewards) for rewards in rewards]).to(device=device)
            next_obs_tensor = torch.stack([next_obs.tensor(normalize=normalize, device="cpu") for next_obs in next_observations]).to(device=device)
            dones_tensor = torch.tensor([float(done) for done in dones]).to(device=device)

            return obs_tensor, actions_tensor, rewards_tensor, next_obs_tensor, dones_tensor
        else:
            return zip(*self.randoms(with_replacement=False).take(size))
    

