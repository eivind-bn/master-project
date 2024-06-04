from . import *

import random

class GenoType:

    @overload
    def __init__(self, 
                 *,   
                 value:             float,
                 volatility:        float,
                 min_value:         float|None = ...,
                 max_value:         float|None = ...,
                 min_volatility:    float|None = ...,
                 max_volatility:    float|None = ...) -> None: ...
        
    @overload
    def __init__(self,    
                 *,  
                 min_value:         float,
                 max_value:         float,
                 min_volatility:    float,
                 max_volatility:    float) -> None: ...
    
    @overload
    def __init__(self,    
                 *,  
                 value:             float,
                 min_value:         float|None = ...,
                 max_value:         float|None = ...,
                 min_volatility:    float,
                 max_volatility:    float) -> None: ...
    
    @overload
    def __init__(self, 
                 *,   
                 volatility:        float,
                 min_value:         float,
                 max_value:         float,
                 min_volatility:    float|None = ...,
                 max_volatility:    float|None = ...) -> None: ...

    def __init__(self,    
                 *,  
                 value:             float|None = None,
                 volatility:        float|None = None,
                 min_value:         float|None = None,
                 max_value:         float|None = None,
                 min_volatility:    float|None = None,
                 max_volatility:    float|None = None) -> None:
        

        if value is None and min_value is not None and max_value is not None:
            self.value = random.random()*(max_value-min_value) + min_value
        elif value is not None:
            self.value = value
        else:
            raise ValueError(f"Constructor requires either value range, the value itself, or both.")

        if min_value is not None:
            assert min_value <= self.value

        if max_value is not None:
            assert max_value >= self.value
            
        if volatility is None and min_volatility is not None and max_volatility is not None:
            self.volatility = random.random()*(max_volatility-min_volatility) + min_volatility
        elif volatility is not None:
            self.volatility = volatility
        else:
            raise ValueError(f"Constructor requires either volatility range, the volatility itself, or both.")

        if min_volatility is not None:
            assert min_volatility <= self.volatility

        if max_volatility is not None:
            assert max_volatility >= self.volatility
        
        self.min_value = min_value
        self.max_value = max_value

        self.min_volatility = min_volatility
        self.max_volatility = max_volatility

    def mutate(self, mutation_rate: float|None = None) -> None:
        if mutation_rate is None or random.random() < mutation_rate:

            self.value += random.normalvariate(sigma=self.volatility)

            if self.max_value is not None and self.min_value is not None:
                while self.value < self.min_value or self.value > self.max_value:
                    if self.value > self.max_value:
                        self.value -= 2*(self.value - self.max_value)
                    elif self.value < self.min_value:
                        self.value -= 2*(self.value - self.min_value)
            elif self.min_value is not None:
                while self.value < self.min_value:
                    self.value -= 2*(self.value - self.min_value)
            elif self.max_value is not None:
                while self.value > self.max_value:
                    self.value -= 2*(self.value - self.max_value)

            self.volatility += random.normalvariate(sigma=self.volatility)

            if self.max_volatility is not None and self.min_volatility is not None:
                while self.volatility < self.min_volatility or self.volatility > self.max_volatility:
                    if self.volatility > self.max_volatility:
                        self.volatility -= 2*(self.volatility - self.max_volatility)
                    elif self.volatility < self.min_volatility:
                        self.volatility -= 2*(self.volatility - self.min_volatility)
            elif self.min_volatility is not None:
                while self.volatility < self.min_volatility:
                    self.volatility -= 2*(self.volatility - self.min_volatility)
            elif self.max_volatility is not None:
                while self.volatility > self.max_volatility:
                    self.volatility -= 2*(self.volatility - self.max_volatility)

    @staticmethod
    def crossover(genotypes: Iterable["GenoType"], weights: Sequence[int|float]|int) -> "GenoType":

        genotypes = iter(genotypes)
        genotype = next(genotypes)

        min_value = genotype.min_value
        max_value = genotype.max_value
        min_volatility = genotype.min_volatility
        max_volatility = genotype.max_volatility

        if isinstance(weights, int):
            weights = Stream(random.random).take(weights).tuple()

        weight_sum = sum(weights)

        if weight_sum == 0.0:
            weight_sum = 1.0

        value = (weights[0]/weight_sum)*genotype.value
        volatility = (weights[0]/weight_sum)*genotype.volatility

        for weight, genotype in zip(weights[1:], genotypes, strict=True):
            assert genotype.min_value == min_value and genotype.max_value == max_value
            assert genotype.min_volatility == min_volatility and genotype.max_volatility == max_volatility

            value += (weight/weight_sum)*genotype.value
            volatility += (weight/weight_sum)*genotype.volatility

        return GenoType(
            value=value,
            volatility=volatility,
            min_value=min_value,
            max_value=max_value,
            min_volatility=min_volatility,
            max_volatility=max_volatility
        )
    
    def __repr__(self) -> str:
        return f"GenoType(value={self.value}, volatility={self.volatility})"