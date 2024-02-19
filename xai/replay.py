from typing import *
from typing import Iterable

from xai.bytes import Memory, GigaBytes
from xai.cache import Cache
from .buffer import Buffer
from .step import Step
from .observation import Observation
from .action import Action
from .reward import Reward

class ReplayBuffer(Buffer[Step]):
    
    def __init__(self, 
                 use_ram: bool, 
                 max_memory: Memory = GigaBytes(5.0), 
                 max_entries: int = 10_000) -> None:
        super().__init__(
            entries=(), 
            eviction_policy="FIFO", 
            use_ram=use_ram, 
            max_memory=max_memory, 
            max_entries=max_entries)
        
    def record(self, 
               observation: Observation, 
               action: Action, 
               rewards: Iterable[Reward], 
               next_observation: Observation, 
               done: bool) -> None:
        self.append(
            entry=Step(
                observation=observation,
                action=action,
                rewards=rewards,
                next_observation=next_observation,
                done=done
            )
        )