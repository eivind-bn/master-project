from typing import *
from .bytes import Memory

import dill
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .buffer import Buffer, EvictionPolicy

X = TypeVar("X", covariant=True)
Y = TypeVar("Y")
Z = TypeVar("Z")

class Stream(Iterable[X], Generic[X]):

    def __init__(self, source: Iterable[X]|Callable[[],X]) -> None:
        super().__init__()
        if isinstance(source, Iterable):
            self._source = source
        else:
            def iterator() -> Iterator[X]:
                while True:
                    yield source()

            self._source = iterator()

    def map(self, f: Callable[[X],Y]) -> "Stream[Y]":
        return Stream(f(x) for x in self)
    
    def zip(self, other: Iterable[Y]) -> "Stream[Tuple[X,Y]]":
        def iterator() -> Iterator[Tuple[X,Y]]:
            for x,y in zip(self,other):
                yield x,y

        return Stream(iterator())
    
    def flatmap(self, f: Callable[[X],Iterable[Y]]) -> "Stream[Y]":
        def iterator() -> Iterator[Y]:
            for x in self:
                for y in f(x):
                    yield y
        
        return Stream(iterator())
    
    def flatten(self: "Stream[Iterable[X]]") -> "Stream[X]":
        return self.flatmap(lambda iterable: iterable)
    
    def enumerate(self) -> "Stream[Tuple[int,X]]":
        def iterator() -> Iterator[Tuple[int,X]]:
            for i,x in enumerate(self):
                yield i,x

        return Stream(iterator())
    
    def filter(self, predicate: Callable[[X],bool]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                if predicate(x):
                    yield x
        
        return Stream(iterator())
    
    def filter_not(self, predicate: Callable[[X],bool]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                if not predicate(x):
                    yield x

        return Stream(iterator())
    
    def take(self, count: int) -> "Stream[X]":
        return (self.enumerate()
                .take_while(lambda ix: ix[0] < count)
                .map(lambda ix: ix[1]))
    
    def take_while(self, predicate: Callable[[X],bool]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                if predicate(x):
                    yield x
                else:
                    break

        return Stream(iterator())
    
    def drop(self, count: int) -> "Stream[X]":
        return (self.enumerate()
                .drop_while(lambda ix: ix[0] < count)
                .map(lambda ix: ix[1]))

    def drop_while(self, predicate: Callable[[X],bool]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            iterator = iter(self)
            for x in iterator:
                if not predicate(x):
                    break

            for x in iterator:
                yield x

        return Stream(iterator())
    
    def range(self, 
              start:    int|None = None, 
              stop:     int|None = None, 
              step:     int = 1) -> "Stream[X]":
        stream = self.enumerate()
        if start is not None:
            stream = stream.drop_while(lambda ix: ix[0] < start)
        if stop is not None:
            stream = stream.take_while(lambda ix: ix[0] < stop)
        if step != 1:
            stream = stream.filter(lambda ix: ix[0] % step == 0)

        return stream.map(lambda ix: ix[1])
    
    def collect(self, *cases: Callable[[X],Tuple[bool,Callable[[],Y]]]) -> "Stream[Y]":
        def iterator() -> Iterator[Y]:
            for x in self:
                for case in cases:
                    cond,result = case(x)
                    if cond:
                        yield result()
                            
        return Stream(iterator())
    
    def reduce(self, start: Y, reducer: Callable[[Y,X],Y]) -> Y:
        y = start
        for x in self:
            y = reducer(y,x)

        return y
    
    def find(self, predicate: Callable[[X],bool], default: Y) -> X|Y:
        for x in self:
            if predicate(x):
                return x
            
        return default
    
    def scan(self, start: Y, scanner: Callable[[Y,X],Y]) -> "Stream[Y]":
        def iterator() -> Iterator[Y]:   
            y = start
            for x in self:
                y = scanner(y,x)
                yield y

        return Stream(iterator())
    
    def peek(self, f: Callable[[X],None]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                f(x)
                yield x

        return Stream(iterator())
    
    def foreach(self, f: Callable[[X],None]) -> None:
        for x in self:
            f(x)

    def chain(self, other: Iterable[X]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                yield x

            for x in other:
                yield x

        return Stream(iterator())
    
    @overload
    def group_by(self:      "Stream[Tuple[Y,Z]]",
                 *,
                 key:       Callable[[X],Tuple[Y,Z]] = ...) -> "Stream[List[Z]]": ...

    @overload
    def group_by(self, 
                 *,
                 key:       Callable[[X],Tuple[Y,Z]]) -> "Stream[List[Z]]": ...

    @overload
    def group_by(self:      "Stream[Tuple[Y,Z]]",
                 *,
                 key:       Callable[[X],Tuple[Y,Z]] = ...,
                 reduce:    Callable[[Z,Z],Z]) -> "Stream[Z]": ...

    @overload
    def group_by(self, 
                 *,
                 key:       Callable[[X],Tuple[Y,Z]], 
                 reduce:    Callable[[Z,Z],Z]) -> "Stream[Z]": ...
        
    def group_by(self, 
                 key:       Callable[[X],Tuple[Y,Z]]|None = None,
                 reduce:    Callable[[Z,Z],Z]|None = None) -> "Stream[List[Z]|Z]":

        return Stream(self.dict(key=key, reduce=reduce)).map(lambda key_value: key_value[1]) # type: ignore

    def all(self, f: Callable[[X],bool]) -> bool:
        return all(f(x) for x in self)
    
    def any(self, f: Callable[[X],bool]) -> bool:
        return any(f(x) for x in self)
    
    def list(self) -> List[X]:
        return list(self)
    
    def tuple(self) -> Tuple[X,...]:
        return tuple(self)
    
    @overload
    def set(self, frozen: Literal[True] = True) -> FrozenSet[X]: ...

    @overload
    def set(self, frozen: Literal[False]) -> Set[X]: ...

    def set(self, frozen: bool = True) -> FrozenSet[X]|Set[X]:
        if frozen:
            return frozenset(self)
        else:
            return set(self)
        
    @overload
    def dict(self:      "Stream[Tuple[Y,Z]]",
             *,
             key:       Callable[[X],Tuple[Y,Z]] = ...) -> Dict[Y,List[Z]]: ...

    @overload
    def dict(self, 
             *,
             key:       Callable[[X],Tuple[Y,Z]]) -> Dict[Y,List[Z]]: ...

    @overload
    def dict(self:      "Stream[Tuple[Y,Z]]",
             *,
             key:       Callable[[X],Tuple[Y,Z]] = ...,
             reduce:    Callable[[Z,Z],Z]) -> Dict[Y,Z]: ...

    @overload
    def dict(self, 
             *,
             key:       Callable[[X],Tuple[Y,Z]], 
             reduce:    Callable[[Z,Z],Z]) -> Dict[Y,Z]: ...
        
    def dict(self, 
             key:       Callable[[X],Tuple[Y,Z]]|None = None,
             reduce:    Callable[[Z,Z],Z]|None = None) -> Mapping[Y,List[Z]|Z]:
        
        if reduce is None:
            key_list_z: Dict[Y, List[Z]] = {}
            if key is None:
                for x in self:
                    if isinstance(x, tuple):
                        y,z = x
                        key_list_z.setdefault(y, []).append(z)
                    else:
                        raise TypeError("Implicit dict construction failed. Expected Stream[Tuple[Y,X]] if key is not provided.")
            else:
                for x in self:
                    y,z = key(x)
                    key_list_z.setdefault(y, []).append(z)

            return key_list_z
        else:
            key_z: Dict[Y,Z] = {}
            if key is None:
                for x in self:
                    if isinstance(x, tuple):
                        y,z2 = x
                        z1 = key_z.get(y)
                        if z1 is None:
                            key_z[y] = z2
                        else:
                            key_z[y] = reduce(z1, z2)
                    else:
                        raise TypeError("Implicit dict construction failed. Expected Stream[Tuple[Y,X]] if key is not provided.")
            else:
                for x in self:
                    y,z2 = key(x)
                    z1 = key_z.get(y)
                    if z1 is None:
                        key_z[y] = z2
                    else:
                        key_z[y] = reduce(z1, z2)

            return key_z
        
    def buffer(self, 
               eviction_policy: "EvictionPolicy", 
               use_ram: bool, 
               max_memory: Memory, 
               max_entries: int, 
               verbose: bool) -> "Buffer[X]":
        from .buffer import Buffer
        return Buffer(
            entries=self,
            eviction_policy=eviction_policy,
            use_ram=use_ram,
            max_memory=max_memory,
            max_entries=max_entries,
            verbose=verbose
        )
    
    def save(self, path: Callable[[X], str]) -> None:
        for x in self:
            with open(path(x), "wb") as file:
                dill.dump(x, file)

    def drain(self) -> None:
        for _ in self:
            pass
    
    def __add__(self, other: Iterable[X]) -> "Stream[X]":
        return self.chain(other)

    def __iter__(self) -> Iterator[X]:
        return iter(self._source)
    
    @staticmethod
    def empty() -> "Stream[Never]":
        return Stream(tuple())