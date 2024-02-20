from typing import *
from tqdm import tqdm
from collections import deque

from .agent import Agent
from .stream import Stream
from .fitness import Fitness
from .buffer import Buffer
from .bytes import Memory
from .asteroids import Asteroids
from .cache import Cache
from .observation import Observation
from .bytes import GigaBytes
from .policy import Policy
from .action import Actions

import random
import multiprocessing as mp
import os
import torch

if TYPE_CHECKING:
    from .genome import Genome

T = TypeVar("T", bound="Genome")

SelectionPolicy = Literal["Elitism", "Random", "Roulette", "Rank"]

class Population(Generic[T]):
    checkpoint_root = "checkpoints"

    def __init__(self, 
                 genomes:                   Iterable[T],
                 use_ram_genomes:           bool,
                 use_ram_observations:      bool|None = None,
                 max_genomes_memory:        Memory|None = GigaBytes(10),
                 max_observations_memory:   Memory|None = GigaBytes(10),
                 verbose:                   bool = True) -> None:

        self._verbose = verbose

        self._genomes = Buffer(
            entries=genomes,
            eviction_policy="Throw", 
            use_ram=use_ram_genomes, 
            max_memory=max_genomes_memory,
            verbose=verbose
        )

        if use_ram_observations is not None:    
            self._observations: Buffer[Observation]|None = Buffer(
                entries=(),
                eviction_policy="Random",
                use_ram=use_ram_observations,
                max_memory=max_observations_memory,
            )
        else:
            self._observations = None
        
    def evolve(self,
               generations:             int,
               survivors_cnt:           int,
               elites_cnt:              int,
               roulettes_cnt:           int,
               random_cnt:              int,
               rank_cnt:                int,
               checkpoints_directory:   str|None,
               number_of_processes:     int,
               save_interval:           int = 3) -> None:
        
        torch.set_num_threads(1)
        
        if checkpoints_directory is not None:
            save_dir = os.path.join(self.checkpoint_root, checkpoints_directory)
            for directory in (self.checkpoint_root, save_dir):
                try:
                    os.mkdir(directory)
                except FileExistsError:
                    continue
        else:
            save_dir = None

        def loader(text: str|None = None) -> Iterator[T]:
            with tqdm(total=self._genomes.entry_size(), desc=text) as bar:
                for genome in self._genomes:
                    yield genome
                    bar.update()

        population_size = self._genomes.entry_size()
        
        with mp.Pool(processes=number_of_processes) as pool:
            for generation in range(generations):

                fitnesses: List[Fitness] = []
                    
                for fitness,observations in pool.imap(self.eval_fitness, loader(f"Generation: {generation}/{generations}")):
                    fitnesses.append(fitness)
                    if self._observations is not None:
                        self._observations.extend(observations)

                weights = Fitness.deviation_score(fitnesses).tuple()

                worst_to_best = Stream(weights).zip(fitnesses).sort(key=lambda wf: wf[0]).map(lambda wf: wf[1]).tuple()
                best_genome_idx = Stream(weights).enumerate().max(None, lambda iw: iw[0])

                if generation % save_interval == 0 and best_genome_idx is not None and save_dir is not None:
                    file_path = os.path.join(save_dir, f"genome{generation}")
                    self._genomes[best_genome_idx[0]].foreach(lambda genome: genome.save(file_path))

                print(f"Worst: {worst_to_best[0]}")
                print(f"Best: {worst_to_best[-1]}")

                survivors_idx = self.selection("Elitism", weights).take(survivors_cnt)

                elites_idx = self.selection("Elitism", weights).take(elites_cnt).tuple()
                roulettes_idx = self.selection("Roulette", weights).take(roulettes_cnt).tuple()
                randoms_idx = self.selection("Random", weights).take(random_cnt).tuple()
                ranks_idx = self.selection("Rank", weights).take(rank_cnt).tuple()

                parents_idx = elites_idx + roulettes_idx + randoms_idx + ranks_idx

                def breed() -> Iterator[T]:
                    while True:
                        p1, p2, = self._genomes[random.choices(parents_idx, k=2)]
                        offspring = p1.__class__(parents=(p1,p2))
                        yield offspring

                self._genomes = self._genomes.new_like((self._genomes[survivors_idx] + breed()).take(population_size))

                assert self._genomes.entry_size() == population_size, f"New population size of {self._genomes.entry_size()} is incorrect."
        
    def observation(self) -> Stream[Observation]:
        if self._observations is not None:
            return self._observations
        else:
            return Stream.empty()

    @staticmethod
    def eval_fitness(genome: T) -> Tuple[Fitness,Tuple[Observation,...]]:
        if "env" not in globals():
            globals()["env"] = Asteroids()

        env: Asteroids = globals()["env"]
        step = 0
        fitness = Fitness()
        recordings: List[Observation] = []
        for episode in range(3):
            observation, rewards = env.reset()
            fitness += Fitness(rewards={reward.name:reward.value for reward in rewards})

            while step < 3000 and env.running():
                step += 1
                actions = genome.predict((observation,))
                for action in actions:
                    observation, rewards = env.step(action)
                    fitness += Fitness(rewards={reward.name:reward.value for reward in rewards})

                    if random.random() < 0.2:
                        recordings.append(observation)

        return fitness, tuple(recordings)

    @overload
    def selection(self, policy: SelectionPolicy, weights: Sequence[int|float]) -> Stream[int]: ...

    @overload
    def selection(self, policy: Literal["Random"]) -> Stream[int]: ...

    def selection(self, policy: SelectionPolicy, weights: Sequence[int|float]|None = None) -> Stream[int]:
        if policy == "Random":
            indices = tuple(range(self._genomes.entry_size()))
            def random_select() -> Iterator[int]:
                while True:
                    index = random.choices(indices)[0]
                    yield index

            return Stream(random_select())
        elif weights is not None:
            assert len(weights) == self._genomes.entry_size()
            match policy:
                case "Elitism":
                    return Stream(sorted(enumerate(weights), key=lambda n: n[1], reverse=True))\
                        .map(lambda ir: ir[0])
                
                case "Roulette":
                    indices = tuple(range(self._genomes.entry_size()))
                    return Stream(lambda: random.choices(indices, weights=weights)[0])
                
                case "Rank":
                    indices = Stream(sorted(enumerate(weights), key=lambda n: n[1], reverse=True))\
                            .map(lambda ir: ir[0])\
                            .tuple()
                    weights = tuple(range(len(indices)))[::-1]
                    return Stream(lambda: random.choices(indices, weights=weights)[0])
                
                case _:
                    assert_never(policy)
        else:
            raise ValueError(f"Selection policy '{policy}' requires weights to be provided.")

    def __add__(self, other: "Population[T]") -> "Population[T]":
        return Population(
            genomes=self._genomes + other._genomes,
            max_genomes_memory=self._genomes.max_memory,
            max_observations_memory=self._observations.max_memory,
            verbose=self._verbose
        )

    def __getitem__(self, loc: int|slice|Iterable[int]) -> Stream[T]:
        return self._genomes[loc]
    
    def __len__(self) -> int:
        return self._genomes.entry_size()