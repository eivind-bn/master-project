from typing import *

from .stream import Stream

class Fitness:

    def __init__(self, 
                 rewards:     Iterable[Tuple[str,float]] = Stream.empty(),
                 penalties:   Iterable[Tuple[str,float]] = Stream.empty()) -> None:
        
        self._rewards = tuple(rewards)
        self._penalties = tuple(penalties)

    def normalized(self, min_fitness: "Fitness", max_fitness: "Fitness") -> "NormalizedFitness":
        normalized_fitness = NormalizedFitness(non_normalized=self)
        for category,reward in self.rewards():
            max, min = max_fitness.get_reward(category), min_fitness.get_reward(category)
            if max - min == 0:
                normalized_fitness.set_reward(category, 1.0)
            else:
                normalized_fitness.set_reward(category, (reward - min)/(max - min))

        for category,penalty in self.penalties():
            max, min = max_fitness.get_penalty(category), min_fitness.get_penalty(category)
            if max - min == 0:
                normalized_fitness.set_penalty(category, 1.0)
            else:
                normalized_fitness.set_penalty(category, (penalty - min)/(max - min))

        return normalized_fitness
    
    def _reduce(self, other: "Fitness", op: Callable[[float,float],float]) -> "Fitness":
        return Fitness(
            rewards=self.rewards().chain(other.rewards()).dict(reduce=lambda z1,z2: op(z1,z2)).items(),
            penalties=self.penalties().chain(other.penalties()).dict(reduce=lambda z1,z2: op(z1,z2)).items()
        )

    def __add__(self, other: "Fitness") -> "Fitness":
        return self._reduce(other, lambda x,y: x+y)
    
    def __sub__(self, other: "Fitness") -> "Fitness":
        return self._reduce(other, lambda x,y: x+y)
    
    def __mul__(self, other: "Fitness") -> "Fitness":
        return self._reduce(other, lambda x,y: x*y)
    
    def __truediv__(self, other: "Fitness") -> "Fitness":
        return self._reduce(other, lambda x,y: x/y)

    def rewards(self) -> Stream[Tuple[str,float]]:
        return Stream(self._rewards)

    def penalties(self) -> Stream[Tuple[str,float]]:
        return Stream(self._penalties)

    def __repr__(self) -> str:
        return str(dict(
            rewards=self._rewards,
            penalties=self._penalties
        ))

    @staticmethod
    def max(fitnesses: Iterable["Fitness"]) -> "Fitness":
        max_fitness = Fitness()
        for fitness in fitnesses:
            for category,reward in fitness.rewards():
                if category in max_fitness._rewards:
                    max_fitness.set_reward(category, max(max_fitness.get_reward(category), reward))
                else:
                    max_fitness.set_reward(category, reward)

            for category,penalty in fitness.penalties():
                if category in max_fitness._penalties:
                    max_fitness.set_penalty(category, max(max_fitness.get_penalty(category), penalty))
                else:
                    max_fitness.set_penalty(category, penalty)

        return max_fitness

    @staticmethod
    def min(fitnesses: Iterable["Fitness"]) -> "Fitness":
        min_fitness = Fitness()
        for fitness in fitnesses:
            for category,reward in fitness.rewards():
                if category in min_fitness._rewards:
                    min_fitness.set_reward(category, min(min_fitness.get_reward(category), reward))
                else:
                    min_fitness.set_reward(category, reward)

            for category,penalty in fitness.penalties():
                if category in min_fitness._penalties:
                    min_fitness.set_penalty(category, min(min_fitness.get_penalty(category), penalty))
                else:
                    min_fitness.set_penalty(category, penalty)

        return min_fitness
    
    @staticmethod
    def normalize(fitnesses: Iterable["Fitness"]) -> Tuple[float,...]:
        fitnesses = tuple(fitnesses)
        min_fitness = Fitness.min(fitnesses)
        max_fitness = Fitness.max(fitnesses)
        return tuple(fitness.normalized(min_fitness=min_fitness, max_fitness=max_fitness) for fitness in fitnesses)
    
    @staticmethod
    def rank(fitnesses: Iterable["Fitness"]) -> Tuple[float,...]:
        pass


class NormalizedFitness(Fitness):

    def __init__(self, 
                 non_normalized: Fitness,
                 rewards: Dict[str, float] | None = None, 
                 penalties: Dict[str, float] | None = None) -> None:
        super().__init__(rewards, penalties)
        self.non_normalized = non_normalized

    def rank(self) -> float:
        reduction = 1.0
        for _,reward in self.rewards():
            reduction *= reward

        for _,penalty in self.penalties():
            reduction *= (1 - penalty)

        return reduction
    
    def un_normalize(self) -> Fitness:
        return self.non_normalized
    
    def __repr__(self) -> str:
        return str({
            "Rank": self.rank(),
            "Rewards": tuple(self.non_normalized.rewards()),
            "Normalized rewards": tuple(self.rewards()),
            "Penalties": tuple(self.non_normalized.penalties()),
            "Normalized penalties": tuple(self.penalties()),
        })