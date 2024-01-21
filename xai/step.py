from typing import *

from .reward import Reward
from .observation import Observation

class Step:

    def __init__(self, observation: Observation, rewards: Sequence[Reward]) -> None:
        self.observation = observation
        self.rewards = tuple(rewards)

    def reward_sum(self) -> int:
        return sum(reward.value for reward in self.rewards)
    
    def reward_group_counts(self) -> Dict[Reward,int]:
        reward_counts: Dict[Reward,int] = {}
        for reward in self.rewards:
            reward_counts[reward] = reward_counts.get(reward, 0) + 1