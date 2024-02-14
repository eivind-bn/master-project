from typing import *
from torch import Tensor
from collections import deque
from .agent import Agent
from .observation import Observation
from .action import Action
from .policy import Policy
from .action import Actions
from .policy import Policy
from .population import Population
from .bytes import Memory
from .buffer import Buffer
from .stream import Stream
from .asteroids import Asteroids

import random

class Genome(Agent):

    @overload
    def __init__(self, 
                 *,
                 parents:   Sequence[Self]) -> None: ...

    @overload
    def __init__(self,
                 *,
                 translate: bool, 
                 rotate:    bool) -> None: ...

    def __init__(self,
                 *,
                 parents:   Sequence[Self]|None = None,
                 translate: bool|None = None, 
                 rotate:    bool|None = None) -> None:
        super().__init__()

        self.stats: Dict[str,Any] = {}
        self._actions = (
            Actions.NOOP,
            Actions.UP,
            Actions.LEFT,
            Actions.RIGHT,
            Actions.FIRE,
        )

        if parents is None:
            assert translate is not None and rotate is not None
            self._translate = translate
            self._rotate = rotate
            self._policy = Policy.new(
                input_dim=Asteroids.observation_shape,
                output_dim=len(self._actions),
                hidden_layers=[2**10, 2**6]
                )
        else:
            self._translate = parents[0]._translate
            self._rotate = parents[0]._rotate

            for parent in parents:
                assert self._translate == parent._translate
                assert self._rotate == parent._rotate

            

                    

    def transform_observation(self, observation: Observation) -> Tensor:
        if self._translate:
            observation = observation.translated()
        if self._rotate:
            observation = observation.rotated()

        return observation.tensor(normalize=True, device="auto")
    
    def predict(self, observation: Observation) -> Action:
        policy = self._policy.predict(self.transform_observation(observation))
        return self._actions[int(policy.tensor(True).argmax().item())]
    
    def populate(self, 
                 population_size:   int,
                 max_memory:        Memory, 
                 verbose:           bool = False) -> Population["Genome"]:
        
        return Population(
            genomes=Stream(lambda: self.clone()).scan(self, lambda y,x: y.breed(x)).take(population_size),
            max_genomes_memory=max_memory,
            verbose=verbose
        )