from . import *
from torch import Tensor
from collections import deque

import random
import torch

class Genome(Agent):

    def __init__(self,
                 translate: bool, 
                 rotate:    bool,
                 memory:    int = 4) -> None:
        super().__init__()

        self._translate = translate
        self._rotate = rotate

        self._network = Network.dense(
            input_dim=(memory,) + Asteroids.observation_shape,
            hidden_layers=[512,256],
            output_dim=(5,),
            output_activation="Softmax"
        )
    
    def rollout(self, env: Asteroids|None = None) -> Stream["Genome.Step"]:
        def iterator() -> Iterator[Genome.Step]:
            nonlocal env
            if env is None:
                env = Asteroids()

            actions = (
                Actions.NOOP,
                Actions.UP,
                Actions.LEFT,
                Actions.RIGHT,
                Actions.FIRE
                )
            observation, rewards = env.reset()
            memory: Deque[Tensor] = deque(maxlen=self._network.input_shape[0])

            def memorize(observation: Observation) -> None:
                if self._translate:
                    observation = observation.translated()
                if self._rotate:
                    observation = observation.rotated()

                tensor = observation.tensor(
                    normalize=True,
                    device=self._network.device
                )

                memory.append(tensor)

            while len(memory) < memory.maxlen:
                if env.running():
                    action = random.choice(actions)
                    last_observation = observation
                    observation, rewards = env.step(action)
                    yield Genome.Step(
                        observation=last_observation,
                        action=action,
                        rewards=rewards,
                        done=not env.running()
                    )
                else:
                    observation, rewards = env.reset()

                memorize(observation)

            while True:
                if env.running():
                    state = torch.stack(tuple(memory))
                    prediction = self._network(state)
                    action_id = int(prediction().argmax().item())
                    action = actions[action_id]
                    last_observation = observation
                    observation, rewards = env.step(action)
                    yield Genome.Step(
                        observation=last_observation,
                        action=action,
                        rewards=rewards,
                        done=not env.running()
                    )
                else:
                    observation, rewards = env.reset()

                memorize(observation)

        return Stream(iterator())
    
    def populate(self, 
                 population_size:           int,
                 location:                  Location,
                 max_memory:                Memory, 
                 genomes_sampling_memsize:  Memory|None = None,
                 observation_sampler:       Callable[[Observation],None]|None = None,
                 verbose:                   bool = False) -> Population["Genome"]:
        
        ref_list = Stream(lambda: self.clone())\
            .scan(self, lambda y,x: Genome(parents=(x,y)))\
            .take(population_size)\
            .ref_list(
                location=location,
                eviction_policy="Throw",
                max_memory=max_memory,
                verbose=verbose
            )
        
        return Population(
            genomes=ref_list,
            observation_sampler=observation_sampler,
            genomes_sampling_memsize=genomes_sampling_memsize
        )