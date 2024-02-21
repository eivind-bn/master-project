from typing import *
from dataclasses import dataclass

from .observation import Observation
from .action import Action
from .reward import Reward

class Step(NamedTuple):
    observation: Observation
    action: Action
    rewards: Tuple[Reward,...]
    next_observation: Observation
    done: bool

    def reward_sum(self) -> int:
        return sum(reward.value for reward in self.rewards)