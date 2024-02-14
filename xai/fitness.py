from typing import *

from .stream import Stream

class Fitness:

    def __init__(self, 
                 rewards:     Dict[str,float]|None = None,
                 penalties:   Dict[str,float]|None = None) -> None:
        
        self._rewards = {} if rewards is None else rewards
        self._penalties = {} if penalties is None else penalties
    
    def _reduce(self, other: "Fitness|int|float", op: Callable[[int|float,int|float],int|float]) -> "Fitness":
        if isinstance(other, Fitness):
            return Fitness(
                rewards=self.rewards().chain(other.rewards()).dict(reduce=lambda z1,z2: op(z1,z2)),
                penalties=self.penalties().chain(other.penalties()).dict(reduce=lambda z1,z2: op(z1,z2))
            )
        else:
            return Fitness(
                rewards=self.rewards().map(lambda kv: (kv[0], op(kv[1], other))).dict(reduce=lambda z1,z2: z1+z2),
                penalties=self.penalties().map(lambda kv: (kv[0], op(kv[1], other))).dict(reduce=lambda z1,z2: z1+z2)
            )

    def __add__(self, other: "Fitness|int|float") -> "Fitness":
        return self._reduce(other, lambda x,y: x+y)
    
    def __sub__(self, other: "Fitness|int|float") -> "Fitness":
        return self._reduce(other, lambda x,y: x-y)
    
    def __mul__(self, other: "Fitness|int|float") -> "Fitness":
        return self._reduce(other, lambda x,y: x*y)
    
    def __truediv__(self, other: "Fitness|int|float") -> "Fitness":
        return self._reduce(other, lambda x,y: x/y)
    
    def __pow__(self, other: "Fitness") -> "Fitness":
        return self._reduce(other, lambda x,y: x**y)
    
    def safe_divide(self, other: "Fitness", neutral_element: float) -> "Fitness":
        return self._reduce(other, lambda x,y: x/y if y != 0.0 else neutral_element)

    def rewards(self) -> Stream[Tuple[str,float]]:
        return Stream(self._rewards.items())

    def penalties(self) -> Stream[Tuple[str,float]]:
        return Stream(self._penalties.items())
    
    def categories(self) -> Stream[Tuple[str,float]]:
        return self.rewards().chain(self.penalties())

    @staticmethod
    def min(fitnesses: Sequence["Fitness"]) -> "Fitness":
        return Fitness(
            rewards=dict(Stream(fitnesses)
                     .flatmap(lambda fitness: fitness.rewards())
                     .group_by(keep_key=True)
                     .map(lambda kr: (kr[0], min(kr[1]) if len(kr[1]) == len(fitnesses) else min(min(kr[1]), 0.0)))),

            penalties=dict(Stream(fitnesses)
                           .flatmap(lambda fitness: fitness.penalties())
                           .group_by(keep_key=True)
                           .map(lambda kp: (kp[0], min(kp[1]) if len(kp[1]) == len(fitnesses) else min(min(kp[1]), 0.0)))),
        )
    
    @staticmethod
    def max(fitnesses: Sequence["Fitness"]) -> "Fitness":
        return Fitness(
            rewards=dict(Stream(fitnesses)
                     .flatmap(lambda fitness: fitness.rewards())
                     .group_by(keep_key=True)
                     .map(lambda kr: (kr[0], max(kr[1]) if len(kr[1]) == len(fitnesses) else max(max(kr[1]), 0.0)))),

            penalties=dict(Stream(fitnesses)
                           .flatmap(lambda fitness: fitness.penalties())
                           .group_by(keep_key=True)
                           .map(lambda kp: (kp[0], max(kp[1]) if len(kp[1]) == len(fitnesses) else max(max(kp[1]), 0.0)))),
        )
    
    @staticmethod
    def normalize(fitnesses: Sequence["Fitness"]) -> Stream["Fitness"]:
        min = Fitness.min(fitnesses)
        max = Fitness.max(fitnesses)
        difference = max - min
        return Stream(fitnesses).map(lambda fitness: (fitness - min).safe_divide(difference, 0.0))
    
    @staticmethod
    def product_score(fitnesses: Sequence["Fitness"]) -> Tuple[float,...]:
        ranks: List[float] = []
        for fitness in Fitness.normalize(fitnesses):
            ranks.append(fitness.rewards().reduce(1.0, lambda z,kv: z*kv[1]) * fitness.penalties().reduce(1.0, lambda z,kv: z*(1.0 - kv[1])))

        return tuple(ranks)

    def __repr__(self) -> str:
        return str(dict(
            rewards=self._rewards,
            penalties=self._penalties
        ))