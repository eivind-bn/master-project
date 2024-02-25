from typing import *
from tqdm import tqdm
from collections import deque

from .agent import Agent
from .stream import Stream
from .fitness import Fitness
from .reflist import RefList
from .bytes import Memory
from .asteroids import Asteroids
from .objref import ObjRef
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
                 genomes:                   RefList[T],
                 observation_sampler:       Callable[[Observation],None]|None = None,
                 genomes_sampling_memsize:  Memory|None = None,
                 verbose:                   bool = True) -> None:

        self._verbose = verbose
        self._genomes_sampling_memsize = genomes_sampling_memsize
        self._observation_sampler = observation_sampler

        self.genomes = genomes
        
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

        def loader(text: str|None = None) -> Iterator[Tuple[T,Memory|None]]:
            if self._genomes_sampling_memsize is not None:
                genome_sampling_memsize = self._genomes_sampling_memsize / number_of_processes
            else:
                genome_sampling_memsize = None

            with tqdm(total=self.genomes.entry_size(), desc=text) as bar:
                for genome in self.genomes:
                    if text is None:
                        ram_used = Memory.ram_used().gigabytes().float()
                        ram_total = Memory.ram_total().gigabytes().float()
                        text = f"Generation: {generation}/{generations}, Ram used: {ram_used/ram_total:.2f}%"

                    yield genome, genome_sampling_memsize

                    bar.set_description(text)
                    bar.update()

        population_size = self.genomes.entry_size()
        
        with mp.Pool(processes=number_of_processes) as pool:
            for generation in range(generations):

                fitnesses: List[Fitness] = []
                    
                for fitness,observations in pool.imap(self.eval_fitness, loader()):
                    fitnesses.append(fitness)
                    if self._observation_sampler is not None:
                        for observation in observations:
                            self._observation_sampler(observation)

                weights = Fitness.deviation_score(fitnesses).tuple()

                worst_to_best = Stream(weights).zip(fitnesses).sort(key=lambda wf: wf[0]).map(lambda wf: wf[1]).tuple()
                best_genome_idx = Stream(weights).enumerate().max(None, lambda iw: iw[0])

                if generation % save_interval == 0 and best_genome_idx is not None and save_dir is not None:
                    file_path = os.path.join(save_dir, f"genome{generation}")
                    self.genomes[best_genome_idx[0]].foreach(lambda genome: genome.save(file_path))

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
                        p1, p2, = self.genomes[random.choices(parents_idx, k=2)]
                        offspring = p1.__class__(parents=(p1,p2))
                        yield offspring

                self.genomes = self.genomes.new_like((self.genomes[survivors_idx] + breed()).take(population_size))

                assert self.genomes.entry_size() == population_size, f"New population size of {self.genomes.entry_size()} is incorrect."

    @staticmethod
    def eval_fitness(genome_and_sampling_memsize: Tuple[T,Memory|None]) -> Tuple[Fitness,Tuple[Observation,...]]:
        if "env" not in globals():
            globals()["env"] = Asteroids()

        env: Asteroids = globals()["env"]
        step = 0
        genome, sampling_memsize = genome_and_sampling_memsize
        fitness = Fitness()

        if sampling_memsize is None:
            recordings: RefList[Observation]|None = None
        else:
            recordings = RefList(
                eviction_policy="Random",
                location="RAM",
                max_memory=sampling_memsize
                )
        
        for episode in range(3):
            observation, rewards = env.reset()
            fitness += Fitness(rewards={reward.name:reward.value for reward in rewards})

            while step < 3000 and env.running():
                step += 1
                actions = genome.predict((observation,))
                for action in actions:
                    observation, rewards = env.step(action)
                    fitness += Fitness(rewards={reward.name:reward.value for reward in rewards})

                    if random.random() < 0.1 and recordings is not None:
                        recordings.append(observation)

        if recordings is not None:
            return fitness, recordings.tuple()
        else:
            return fitness, tuple()

    @overload
    def selection(self, policy: SelectionPolicy, weights: Sequence[int|float]) -> Stream[int]: ...

    @overload
    def selection(self, policy: Literal["Random"]) -> Stream[int]: ...

    def selection(self, policy: SelectionPolicy, weights: Sequence[int|float]|None = None) -> Stream[int]:
        if policy == "Random":
            indices = tuple(range(self.genomes.entry_size()))
            def random_select() -> Iterator[int]:
                while True:
                    index = random.choices(indices)[0]
                    yield index

            return Stream(random_select())
        elif weights is not None:
            assert len(weights) == self.genomes.entry_size()
            match policy:
                case "Elitism":
                    return Stream(sorted(enumerate(weights), key=lambda n: n[1], reverse=True))\
                        .map(lambda ir: ir[0])
                
                case "Roulette":
                    indices = tuple(range(self.genomes.entry_size()))
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
            genomes=self.genomes + other.genomes,
            observation_sampler=self._observation_sampler,
            genomes_sampling_memsize=self._genomes_sampling_memsize,
            verbose=self._verbose
        )

    def __getitem__(self, loc: int|slice|Iterable[int]) -> Stream[T]:
        return self.genomes[loc]
    
    def __len__(self) -> int:
        return self.genomes.entry_size()