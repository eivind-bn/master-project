from typing import *

class Fitness:

    def __init__(self, 
                 rewards:     Dict[str,float]|None = None,
                 penalties:   Dict[str,float]|None = None) -> None:
        
        if rewards is None:
            self._named_rewards: Dict[str,float] = {}
        else:
            self._named_rewards = rewards.copy()

        if penalties is None:
            self._named_penalties: Dict[str,float] = {}
        else:
            self._named_penalties = penalties.copy()

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
    
    def _reduce(self, operand: "Fitness", operation: Callable[[float,float],float]) -> "Fitness":
        result = self.copy()
        for category,reward in operand.rewards():
            result._named_rewards[category] = result._named_rewards.get(category, 0) + reward

        for category,penalty in operand.penalties():
            result._named_penalties[category] = result._named_penalties.get(category, 0) + penalty

        return result

    def __add__(self, other: "Fitness") -> "Fitness":
        result = self.copy()
        for category,reward in other.rewards():
            result._named_rewards[category] = result._named_rewards.get(category, 0) + reward

        for category,penalty in other.penalties():
            result._named_penalties[category] = result._named_penalties.get(category, 0) + penalty

        return result
    
    def __sub__(self, other: "Fitness") -> "Fitness":
        result = self.copy()
        for category,reward in other.rewards():
            result._named_rewards[category] = result._named_rewards.get(category, 0) + reward

        for category,penalty in other.penalties():
            result._named_penalties[category] = result._named_penalties.get(category, 0) + penalty

        return result

    def rewards(self) -> Iterator[Tuple[str,float]]:
        for category,reward in self._named_rewards.items():
            yield category,reward

    def penalties(self) -> Iterator[Tuple[str,float]]:
        for category,penalty in self._named_penalties.items():
            yield category,penalty
    
    def get_reward(self, category: str) -> float:
        return self._named_rewards.get(category, 0)
    
    def set_reward(self, category: str, reward: float) -> None:
        self._named_rewards[category] = reward
    
    def get_penalty(self, category: str) -> float:
        return self._named_penalties.get(category, 0)
    
    def set_penalty(self, category: str, penalty: float) -> None:
        self._named_penalties[category] = penalty

    def __repr__(self) -> str:
        return str(dict(
            rewards=self._named_rewards,
            penalties=self._named_penalties
        ))

    def copy(self) -> "Fitness":
        return Fitness(
            rewards=self._named_rewards.copy(),
            penalties=self._named_penalties.copy()
        )

    @staticmethod
    def max(fitnesses: Iterable["Fitness"]) -> "Fitness":
        max_fitness = Fitness()
        for fitness in fitnesses:
            for category,reward in fitness.rewards():
                if category in max_fitness._named_rewards:
                    max_fitness.set_reward(category, max(max_fitness.get_reward(category), reward))
                else:
                    max_fitness.set_reward(category, reward)

            for category,penalty in fitness.penalties():
                if category in max_fitness._named_penalties:
                    max_fitness.set_penalty(category, max(max_fitness.get_penalty(category), penalty))
                else:
                    max_fitness.set_penalty(category, penalty)

        return max_fitness

    @staticmethod
    def min(fitnesses: Iterable["Fitness"]) -> "Fitness":
        min_fitness = Fitness()
        for fitness in fitnesses:
            for category,reward in fitness.rewards():
                if category in min_fitness._named_rewards:
                    min_fitness.set_reward(category, min(min_fitness.get_reward(category), reward))
                else:
                    min_fitness.set_reward(category, reward)

            for category,penalty in fitness.penalties():
                if category in min_fitness._named_penalties:
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