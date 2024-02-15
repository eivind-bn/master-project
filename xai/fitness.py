from typing import *

from .stream import Stream

class Fitness:

    def __init__(self, 
                 rewards:     Dict[str,float]|None = None,
                 penalties:   Dict[str,float]|None = None) -> None:
        
        self._rewards = {} if rewards is None else rewards
        self._penalties = {} if penalties is None else penalties

    def _compute(self,
                 operands:      "Iterable[Fitness|int|float]", 
                 operator:      Callable[[int|float,int|float],int|float],
                 inplace:       bool = False) -> "Fitness":
        
        result = self if inplace else self.copy()

        for operand in operands:
            if isinstance(operand, Fitness):
                for key,value in operand.rewards():
                    result._rewards[key] = operator(result._rewards.get(key, 0.0), value)

                for key,value in operand.penalties():
                    result._penalties[key] = operator(result._penalties.get(key, 0.0), value)
            else:
                for key,value in result.rewards():
                    result._rewards[key] = operator(value, operand)

                for key,value in result.penalties():
                    result._penalties[key] = operator(value, operand)

        return result

    def __add__(self, operand: "Fitness|int|float") -> "Fitness":
        return self._compute((operand,), lambda x,y: x+y)
    
    def __sub__(self, operand: "Fitness|int|float") -> "Fitness":
        return self._compute((operand,), lambda x,y: x-y)
    
    def __mul__(self, operand: "Fitness|int|float") -> "Fitness":
        return self._compute((operand,), lambda x,y: x*y)
    
    def __truediv__(self, operand: "Fitness|int|float") -> "Fitness":
        return self._compute((operand,), lambda x,y: x/y)
    
    def __pow__(self, operand: "Fitness|int|float") -> "Fitness":
        return self._compute((operand,), lambda x,y: x**y)
    
    def safe_divide(self, operand: "Fitness", neutral_element: float) -> "Fitness":
        return self._compute((operand,), lambda x,y: x/y if y != 0.0 else neutral_element)

    def rewards(self) -> Stream[Tuple[str,float]]:
        return Stream(self._rewards.items())

    def penalties(self) -> Stream[Tuple[str,float]]:
        return Stream(self._penalties.items())
    
    def categories(self) -> Stream[Tuple[str,float]]:
        return self.rewards().chain(self.penalties())
    
    def copy(self) -> "Fitness":
        return Fitness(
            rewards=self._rewards.copy(),
            penalties=self._penalties.copy()
        )

    @staticmethod
    def min(fitnesses: Sequence["Fitness"]) -> "Fitness":
        if fitnesses:
            return Fitness._compute(fitnesses[0], fitnesses[1:], lambda x,y: min(x,y))
        else:
            return Fitness()
    
    @staticmethod
    def max(fitnesses: Sequence["Fitness"]) -> "Fitness":
        if fitnesses:
            return Fitness._compute(fitnesses[0], fitnesses[1:], lambda x,y: max(x,y))
        else:
            return Fitness()
    
    @staticmethod
    def normalize(fitnesses: Sequence["Fitness"]) -> Stream["Fitness"]:
        min = Fitness.min(fitnesses)
        max = Fitness.max(fitnesses)
        difference = max - min
        return Stream(fitnesses).map(lambda fitness: (fitness - min).safe_divide(difference, 0.0))
    
    @staticmethod
    def product_score(fitnesses: Sequence["Fitness"]) -> Stream[float]:
        def iterator() -> Iterator[float]:
            for fitness in Fitness.normalize(fitnesses):
                reward_score = fitness.rewards().reduce(1.0, lambda z,kv: z*kv[1])
                penalty_score = fitness.penalties().reduce(1.0, lambda z,kv: z*(1.0 - kv[1]))
                yield reward_score * penalty_score

        return Stream(iterator())
    
    @staticmethod
    def deviation_score(fitnesses: Sequence["Fitness"]) -> Stream[float]:
        def iterator() -> Iterator[float]:
            deviations: List[float] = []
            for fitness in Fitness.normalize(fitnesses):
                reward = fitness.rewards().reduce(0.0, lambda y,x: y+(1-x[1])**2)
                penalty = fitness.penalties().reduce(0.0, lambda y,x: y+x[1]**2)
                deviations.append(abs(reward + penalty))

            min,max = Stream(deviations).min_max(default=0.0)

            difference = max - min
            if difference == 0.0:
                for score in deviations:
                    yield 1 - (score - min)
            else:
                for score in deviations:
                    yield 1 - ((score - min)/difference)

        return Stream(iterator())

    def __repr__(self) -> str:
        return str(dict(
            rewards=self._rewards,
            penalties=self._penalties
        ))