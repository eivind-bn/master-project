from typing import *
from typing import Iterator

X = TypeVar("X")
Y = TypeVar("Y")

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
        return self.filter(lambda x: not predicate(x))
    
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
    
    def collect(self, collector: Dict[X,Y]) -> "Stream[Y]":
        def iterator() -> Iterator[Y]:
            for x in self:
                y = collector.get(x)
                if y is not None:
                    yield y

        return Stream(iterator())
    
    def reduce(self, reducer: Callable[[Y,X],Y], start: Y) -> Y:
        y = start
        for x in self:
            y = reducer(y,x)

        return y
    
    def peek(self, f: Callable[[X],None]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                f(x)
                yield x

        return Stream(iterator())
    
    def foreach(self, f: Callable[[X],None]) -> None:
        for x in self:
            f(x)

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

    def __iter__(self) -> Iterator[X]:
        return iter(self._source)