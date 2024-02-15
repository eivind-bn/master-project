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
import torch

class Genome(Agent):

    @overload
    def __init__(self, 
                 *,
                 parents:   Sequence[Self],
                 weights:   Sequence[int|float] = ...) -> None: ...

    @overload
    def __init__(self,
                 *,
                 translate: bool, 
                 rotate:    bool) -> None: ...

    def __init__(self,
                 *,
                 parents:   Sequence[Self]|None = None,
                 translate: bool|None = None, 
                 rotate:    bool|None = None,
                 weights:   Sequence[int|float]|None = None) -> None:
        super().__init__()

        self.stats: Dict[str,Any] = {}
        self._memory: Deque[Tensor] = deque(maxlen=4)
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
            self._head = Policy.new(
                input_dim=Asteroids.observation_shape,
                output_dim=64,
                hidden_layers=0,
                device="cpu"
                )
            self._tail = Policy.new(
                input_dim=(4,) + self._head.output_dim,
                output_dim=len(self._actions),
                device="cpu"
                )
        else:
            self._translate = parents[0]._translate
            self._rotate = parents[0]._rotate

            parent_heads = Stream(parents).map(lambda p: p._head).tuple()
            parent_tails = Stream(parents).map(lambda p: p._tail).tuple()
            
            if weights is None:
                weights = Stream(random.random).take(len(parent_tails)).tuple()

            self._head = Policy.crossover(parent_heads, weights)
            self._tail = Policy.crossover(parent_tails, weights)
            
            self._head.mutate(0.4, rate=None)
            self._tail.mutate(0.4, rate=None) 

    def transform_observation(self, observation: Observation) -> Tensor:
        if self._translate:
            observation = observation.translated()
        if self._rotate:
            observation = observation.rotated()

        return observation.tensor(normalize=True, device="cpu")
    
    def predict(self, observations: Sequence[Observation]) -> Sequence[Action]:
        actions: List[Action] = []
        for observation in observations:
            tensor = self.transform_observation(observation)
            self._memory.append(self._head.predict(tensor).tensor(True))
            if len(self._memory) < 4:
                continue

            policy = self._tail.predict(torch.stack(tuple(self._memory))).tensor(True)
            action = self._actions[int(policy.argmax().item())]
            actions.append(action)

        return tuple(actions)
    
    def populate(self, 
                 population_size:   int,
                 max_memory:        Memory, 
                 verbose:           bool = False) -> Population["Genome"]:
        
        return Population(
            genomes=Stream(lambda: self.clone()).scan(self, lambda y,x: Genome(parents=(x,y))).take(population_size),
            max_genomes_memory=max_memory,
            verbose=verbose
        )