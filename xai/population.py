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

class Population(Generic[T], Sequence[Cache[T]]):
    checkpoint_root = "checkpoints"

    def __init__(self, 
                 genomes:       Iterable[Cache[T]|T],
                 max_memory:    Memory,
                 verbose:       bool = True) -> None:

        self._max_memory = max_memory
        self._verbose = verbose
        self._fitnesses: List[NormalizedFitness] = []
        self._genome_pool = Buffer(
            entries=genomes,
            eviction_policy="Random", 
            use_ram=False, 
            max_memory=self._max_memory,
            verbose=verbose
            )
        
        self._observation_pool: Buffer[Observation] = Buffer(
            entries=(),
            eviction_policy="Random",
            use_ram=False,
            max_memory=GigaBytes(5),
        )
        
    def evolve(self,
               generations: int,
               survivors_cnt: int,
               elites_cnt: int,
               roulettes_cnt: int,
               random_cnt: int,
               checkpoints_directory: str|None,
               number_of_processes: int) -> None:
        
        torch.set_num_threads(1)
        
        if checkpoints_directory is not None:
            save_dir = os.path.join(self.checkpoint_root, checkpoints_directory)
            for directory in (self.checkpoint_root, save_dir):
                try:
                    os.mkdir(directory)
                except FileExistsError:
                    continue

        def loader(text: str|None = None) -> Iterator[T]:
            with tqdm(total=self._genome_pool.length(), desc=text) as bar:
                for cache in self._genome_pool:
                    with cache as genome:
                        yield genome
                    bar.update()
        
        with mp.Pool(processes=number_of_processes) as pool:
            for generation in range(generations):


                self._fitnesses.clear()
                encoder: Policy
                decoder: Policy
                autoencoder: Policy
                    
                for genome in pool.imap(self.eval_fitness, loader(f"Generation: {generation}/{generations}")):
                    encoder = genome.encoder
                    decoder = genome.decoder
                    autoencoder = genome.autoencoder
                    self._fitnesses.append(genome.fitness)
                    self._observation_pool = self._observation_pool.extended(genome.observations)

                observations = []
                for cache in self._observation_pool.take(500):
                    with cache as observation:
                        observations.append(observation.translated().rotated().tensor(True, "auto"))

                observations = torch.stack(observations)
                autoencoder.adam().fit(
                    X=observations,
                    Y=observations,
                    epochs=100,
                    batch_size=128,
                    loss_criterion="MSELoss",
                    verbose=True
                )

                for cache in self._genome_pool:
                    with cache as genome:
                        genome.encoder = encoder
                        genome.decoder = decoder
                        genome.autoencoder = autoencoder

                rewards: List[float] = []

                for fitness in self._fitnesses:
                    total_reward = 0

                    for name,reward in fitness.rewards():
                        total_reward += reward

                    rewards.append(total_reward)

                print(f"Max: {max(rewards)} Min: {min(rewards)}, Mean: {sum(rewards)/len(rewards)}")
                


                    


                survivors = self.selection("Elitism").take(survivors_cnt).tuple()

                elites = self.selection("Elitism").take(elites_cnt).tuple()
                roulettes = self.selection("Roulette").take(roulettes_cnt).tuple()
                randoms = self.selection("Random").take(random_cnt).tuple()

                offsprings = self.populate(elites + roulettes + randoms)\
                    .take(self._genome_pool.length()-survivors_cnt)\
                    .tuple()
                
                assert len(offsprings) + 1 == self._genome_pool.length()

                self._genome_pool = Buffer(
                    entries=offsprings + survivors,
                    eviction_policy="Random",
                    use_ram=False,
                    max_memory=self._max_memory,
                    verbose=self._verbose
                )

    def populate(self, parents: Sequence[Cache[T]]) -> "Stream[T]":
        assert len(parents) > 0

        def iterator() -> Iterator[T]:
            while True:
                s1,s2 = random.choices(parents, k=2)
                with s1 as p1, s2 as p2:
                    offspring = p1.breed([p2], volatility=0.05, mutation_rate=0.05)
                    yield offspring

        return Stream(iterator())
        
    @staticmethod
    def eval_fitness(genome: T) -> T:
        if "env" not in globals():
            globals()["env"] = Asteroids()

        env: Asteroids = globals()["env"]
        step = 0
        fitness = Fitness(rewards={"game_score": 0.0})
        for episode in range(3):
            observation, rewards = env.reset()
            while step < 3000 and env.running():
                step += 1
                action = genome.predict(observation)
                observation, rewards = env.step(action)
                if random.random() < 0.2:
                    genome.observations.append(observation)
                fitness += Fitness(rewards={"game_score": sum(reward.value for reward in rewards)})

        genome.fitness = fitness
        return genome

    @overload
    def selection(self, 
                  policy:   Literal["Elitism", "Roulette"], 
                  criteria: Callable[[T],int|float]|None = None) -> Stream[Cache[T]]: ...

    @overload
    def selection(self, policy: SelectionPolicy) -> Stream[Cache[T]]: ...

    def selection(self, policy: SelectionPolicy, criteria: Callable[[T],int|float]|None = None) -> Stream[Cache[T]]:
        if self._fitnesses is None:
            return Stream(self._genome_pool)
        
        ranks = tuple(fitness.rank() for fitness in Fitness.normalize_all(self._fitnesses))
        indexed_ranks = sorted(enumerate(ranks), key=lambda n: n[1], reverse=True)

        match policy:
            case "Elitism":
                return Stream(indexed_ranks).map(lambda index_rank: self._genome_pool[index_rank[0]])
            case "Roulette":
                def roulette_select() -> Iterator[Cache[T]]:
                    while True:
                        index,_ = random.choices(indexed_ranks, weights=ranks)[0]
                        yield self._genome_pool[index]

                return Stream(roulette_select())
            case "Random":
                def random_select() -> Iterator[Cache[T]]:
                    while True:
                        index,_ = random.choices(indexed_ranks)[0]
                        yield self._genome_pool[index]

                return Stream(random_select())
            case _:
                assert_never(policy)

    def __add__(self, other: "Population[T]") -> "Population[T]":
        return Population(
            max_memory=self._max_memory,
            genomes=self._genome_pool + other._genome_pool,
            verbose=False
        )

    @overload
    def __getitem__(self, loc: int) -> Cache[T]: ...
    
    @overload
    def __getitem__(self, loc: slice) -> Tuple[Cache[T],...]: ...

    def __getitem__(self, loc: int|slice) -> Cache[T]|Tuple[Cache[T],...]:
        return self._genome_pool[loc]
    
    def __len__(self) -> int:
        return len(self._genome_pool)