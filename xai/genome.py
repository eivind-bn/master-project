from typing import *
from torch import Tensor
from collections import deque
from .agent import Agent
from .observation import Observation
from .action import Action
from .policy import Policy
from .action import Actions
from .policy import Policy
from .fitness import NormalizedFitness
from .population import Population
from .bytes import Memory
from .buffer import Buffer
from .stream import Stream

import random

class Genome(Agent):

    def __init__(self, 
                 encoder: Policy, 
                 decoder: Policy, 
                 translate: bool, 
                 rotate: bool,
                 policy: Policy|None = None) -> None:
        super().__init__()
        self.stats: Dict[str,Any] = {}
        self._translate = translate
        self._rotate = rotate
        self._actions = (
            Actions.NOOP,
            Actions.UP,
            Actions.LEFT,
            Actions.RIGHT,
            Actions.FIRE,
        )
        self.encoder = encoder
        self.decoder = decoder

        if policy is not None:
            self.policy = policy
        else:
            self.policy = Policy.new(
                input_dim=self.encoder.output_dim,
                output_dim=len(self._actions),
                hidden_activation="ReLU"
            )
        self.autoencoder = self.encoder + self.decoder

    def transform_observation(self, observation: Observation) -> Tensor:
        if self._translate:
            observation = observation.translated()
        if self._rotate:
            observation = observation.rotated()

        return observation.tensor(normalize=True, device="auto")
    
    def predict(self, observation: Observation) -> Action:
        latent = self.encoder.predict(self.transform_observation(observation))
        policy = self.policy.predict(latent).numpy()
        return self._actions[policy.argmax()]

    def breed(self, 
              partners:         Iterable[Self],
              volatility:       float,
              mutation_rate:    float) -> "Genome":
        policies = (self.policy,) + tuple(partner.policy for partner in partners)
        fitnesses = tuple(genome.fitness for genome in (self,) + tuple(partners))
        policy = self.policy.crossover(policies, fitnesses)
        self.policy.mutate(volatility=volatility, rate=mutation_rate)
        return Genome(
            encoder=self.encoder,
            decoder=self.decoder,
            translate=self._translate,
            rotate=self._rotate,
            policy=policy
        )
    
    def populate(self, 
                 population_size:   int,
                 max_memory:        Memory, 
                 verbose:           bool = False) -> Population["Genome"]:
        
        return Population(
            genomes=Stream(lambda: self.clone()).scan(self, lambda y,x: y.breed(x)).take(population_size),
            max_genomes_memory=max_memory,
            verbose=verbose
        )