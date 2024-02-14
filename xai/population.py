from typing import *
from tqdm import tqdm

from .agent import Agent
from .stream import Stream
from .fitness import NormalizedFitness, Fitness
from .buffer import Buffer
from .bytes import Memory
from .asteroids import Asteroids
from .cache import Cache
from .observation import Observation
from .bytes import GigaBytes
from .policy import Policy

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
                 max_genomes_memory:        Memory|None = GigaBytes(10),
                 max_observations_memory:   Memory|None = GigaBytes(10),
                 verbose:                   bool = True) -> None:

        self._verbose = verbose

        self._genomes = Buffer(
            entries=genomes,
            eviction_policy="Throw", 
            use_ram=False, 
            max_memory=max_genomes_memory,
            verbose=verbose
        )
        
        self._observations: Buffer[Observation] = Buffer(
            entries=(),
            eviction_policy="Random",
            use_ram=False,
            max_memory=max_observations_memory,
        )
        
    def evolve(self,
               generations:             int,
               survivors_cnt:           int,
               elites_cnt:              int,
               roulettes_cnt:           int,
               random_cnt:              int,
               rank_cnt:                int,
               checkpoints_directory:   str|None,
               number_of_processes:     int) -> None:
        
        torch.set_num_threads(1)
        
        if checkpoints_directory is not None:
            save_dir = os.path.join(self.checkpoint_root, checkpoints_directory)
            for directory in (self.checkpoint_root, save_dir):
                try:
                    os.mkdir(directory)
                except FileExistsError:
                    continue

        def loader(text: str|None = None) -> Iterator[T]:
            with tqdm(total=self._genomes.entry_size(), desc=text) as bar:
                for genome in self._genomes:
                    yield genome
                    bar.update()
        
        with mp.Pool(processes=number_of_processes) as pool:
            for generation in range(generations):

                fitnesses: List[Fitness] = []
                    
                for fitness,observations in pool.imap(self.eval_fitness, loader(f"Generation: {generation}/{generations}")):
                    fitnesses.append(fitness)
                    self._observations = self._observations.extended(observations)

                weights = Fitness.product_score(fitnesses)

                rankings = Stream(weights).enumerate().sort(key=lambda t: t[1]).tuple()
                if rankings:
                    worst = rankings[0]
                    best = rankings[1]

                    print(f"Worst: {worst}")
                    print(f"Best: {best}")

                survivors_idx = self.selection("Elitism", weights).take(survivors_cnt)

                elites_idx = self.selection("Elitism", weights).take(elites_cnt).tuple()
                roulettes_idx = self.selection("Roulette", weights).take(roulettes_cnt).tuple()
                randoms_idx = self.selection("Random", weights).take(random_cnt).tuple()
                ranks_idx = self.selection("Rank", weights).take(rank_cnt).tuple()

                parents_idx = elites_idx + roulettes_idx + randoms_idx + ranks_idx

                def breed() -> Iterator[T]:
                    p1, p2, = self._genomes[random.choices(parents_idx, k=2)]
                    offspring = p1.breed(p2)
                    yield offspring

                old_population_idx = range(self._genomes.entry_size())

                self._genomes.extended(
                    other=self._genomes[survivors_idx] + Stream(breed()).take(len(old_population_idx)),
                    eviction_policy="Throw",
                    verbose=True
                )

                self._genomes.remove(old_population_idx)
        
    @staticmethod
    def eval_fitness(genome: T) -> Tuple[Fitness,Tuple[Observation,...]]:
        if "env" not in globals():
            globals()["env"] = Asteroids()

        env: Asteroids = globals()["env"]
        step = 0
        fitness = Fitness()
        observations: List[Observation] = []
        for episode in range(3):
            observation, rewards = env.reset()
            while step < 3000 and env.running():
                step += 1
                action = genome.predict(observation)
                observation, rewards = env.step(action)
                if random.random() < 0.2:
                    observations.append(observation)
                fitness += Fitness(rewards={"game_score": sum(reward.value for reward in rewards)})

        return fitness, tuple(observations)

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