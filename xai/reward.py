from typing import *
from dataclasses import dataclass

@dataclass
class Reward:
    values: List[int|float]
    observations: List[O]
    actions: List[A]

    def native_game_reward(self) -> int|float:
        return sum(self.values)
    
    def __add__(self, other: "Reward[O,A]") -> "Reward[O,A]":
        return Reward(
            values=self.values + other.values,
            observations=self.observations + other.observations,
            actions=self.actions
        )
    
    def __iadd__(self, other: "Reward[O,A]") -> None:
        self.values.extend(other.values)
        self.observations.extend(other.observations)
        self.actions.extend(other.actions)