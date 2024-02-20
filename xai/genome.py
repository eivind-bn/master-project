from typing import *
from dataclasses import dataclass
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
from .genotype import GenoType

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
                 translate:         bool, 
                 rotate:            bool,
                 volatility:        float,
                 mutation_rate:     float = ...,
                 head_output_dim:   int = ...) -> None: ...

    def __init__(self,
                 *,
                 parents:           Sequence[Self]|None = None,
                 translate:         bool|None = None, 
                 rotate:            bool|None = None,
                 weights:           Sequence[int|float]|None = None,
                 volatility:        float|None = None,
                 mutation_rate:     float = 1.0,
                 head_output_dim:   int = 64) -> None:
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
            assert translate is not None and rotate is not None and volatility is not None
            self._translate = translate
            self._rotate = rotate

            self._mutation_rate = GenoType(
                value=mutation_rate,
                min_value=0.0,
                max_value=1.0,
                volatility=1.0,
                min_volatility=0.0
            )

            self._volatility = GenoType(
                value=volatility,
                min_value=0.0,
                volatility=1.0,
                min_volatility=0.0
            )

            self._head = Policy.new(
                input_dim=Asteroids.observation_shape,
                output_dim=head_output_dim,
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
            parent_mutation_rates = Stream(parents).map(lambda p: p._mutation_rate).tuple()
            parent_volatilities = Stream(parents).map(lambda p: p._volatility).tuple()
            
            if weights is None:
                weights = Stream(random.random).take(len(parent_tails)).tuple()

            self._head = Policy.crossover(parent_heads, weights)
            self._tail = Policy.crossover(parent_tails, weights)
            self._mutation_rate = GenoType.crossover(parent_mutation_rates, weights)
            self._volatility = GenoType.crossover(parent_volatilities, weights)

            self._mutation_rate.mutate(self._mutation_rate.value)
            self._volatility.mutate(self._mutation_rate.value)
            self._head.mutate(self._volatility.value, self._mutation_rate.value)
            self._tail.mutate(self._volatility.value, self._mutation_rate.value) 

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
                 population_size:           int,
                 use_ram_genomes:           bool,
                 use_ram_observations:      bool|None,
                 max_memory:                Memory, 
                 max_observation_memory:    Memory,
                 verbose:                   bool = False) -> Population["Genome"]:
        
        return Population(
            genomes=Stream(lambda: self.clone()).scan(self, lambda y,x: Genome(parents=(x,y))).take(population_size),
            max_genomes_memory=max_memory,
            max_observations_memory=max_observation_memory,
            use_ram_genomes=use_ram_genomes,
            use_ram_observations=use_ram_observations,
            verbose=verbose
        )