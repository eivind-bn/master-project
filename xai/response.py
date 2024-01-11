from typing import *

from reward import Reward
from observation import Observation

class Response:

    def __init__(self, observation_reward_tuples: Iterable[Tuple[Observation,Reward]]) -> None:
        self._observation_reward_tuples = tuple(observation_reward_tuples)

    def __getitem__(self, index: int) -> Tuple[Observation,Reward]:
        return self._observation_reward_tuples[index]
    
    def __iter__(self) -> Iterator[Tuple[Observation,Reward]]:
        return iter(self._observation_reward_tuples)

