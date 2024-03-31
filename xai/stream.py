from . import *

import dill # type: ignore
import random

from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

if TYPE_CHECKING:
    from .reflist import RefList, EvictionPolicy, Location

X = TypeVar("X", covariant=True)
Y = TypeVar("Y")
Z = TypeVar("Z")

@dataclass
class _DillPickle(Generic[Y,Z]):
    f: Callable[[Y],Z]

    def __getstate__(self) -> bytes:
        return cast(bytes, dill.dumps(self.f))
    
    def __setstate__(self, dump: bytes) -> None:
        self.f = dill.loads(dump)
    
    def __call__(self, x: Y) -> Z:
        return self.f(x)

class Stream(Iterable[X]):

    def __init__(self, 
                 source:        Iterable[X]|Callable[[],X], 
                 expected_size: int|None = None) -> None:
        super().__init__()
        if isinstance(source, Iterable):
            self._source = source
        else:
            def iterator() -> Iterator[X]:
                while True:
                    yield source()

            self._source = iterator()

        self._expected_length: int|None = None

        if expected_size is None:
            if isinstance(source, Sized):
                self._expected_length = len(source)
            elif isinstance(source, Stream):
                self._expected_length = source._expected_length
        else:
            self._expected_length = expected_size

    def map(self, f: Callable[[X],Y]) -> "Stream[Y]":
        return Stream((f(x) for x in self), expected_size=self._expected_length)
    
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

        return Stream(iterator(), expected_size=self._expected_length)
    
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
        def iterator() -> Iterator[X]:
            stop = count - 1
            for i,x in enumerate(self):
                yield x
                if i == stop:
                    break

        if self._expected_length is None:
            expected_length = None
        else:
            expected_length = min(self._expected_length, count)
        
        return Stream(iterator(), expected_size=expected_length)
    
    def take_while(self, predicate: Callable[[X],bool]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                if predicate(x):
                    yield x
                else:
                    break

        return Stream(iterator())
    
    def drop(self, count: int) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            iterator = iter(self)
            start = count - 1

            for i,x in enumerate(iterator):
                if i == start:
                    break

            for x in iterator:
                yield x

        if self._expected_length is None:
            expected_length = None
        else:
            expected_length = max(self._expected_length-count, 0)

        return Stream(iterator(), expected_size=expected_length)

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
                        break
                            
        return Stream(iterator())
    
    def reduce(self, start: Y, reducer: Callable[[Y,X],Y]) -> Y:
        y = start
        for x in self:
            y = reducer(y,x)

        return y
    
    def sum(self: "Stream[int|float]") -> float:
        return self.reduce(0.0, lambda y,x: y+x)
    
    def prod(self: "Stream[int|float]") -> float:
        return self.reduce(1.0, lambda y,x: y*x)
    
    def find(self, predicate: Callable[[X],bool], default: Y) -> X|Y:
        for x in self:
            if predicate(x):
                return x
            
        return default
    
    @overload
    def sort(self:      "Stream[int]",
             key:       Callable[[X],int|float] = ...,
             ascending: bool = ...) -> "Stream[X]": ...
    
    @overload
    def sort(self:      "Stream[float]",
             key:       Callable[[X],int|float] = ...,
             ascending: bool = ...) -> "Stream[X]": ...
    
    @overload
    def sort(self,
             key:       Callable[[X],int|float],
             ascending: bool = ...) -> "Stream[X]": ...
    
    def sort(self, 
             key:       Callable[[X],int|float]|None = None,
             ascending: bool = True) -> "Stream[X]":
        
        elements = self.list()

        if elements:
            if key is None:
                if isinstance(elements[0], (int,float)):
                    number_elements = cast(List[int|float], elements)
                    number_stream = Stream(sorted(number_elements, reverse=not ascending), expected_size=len(number_elements))
                    return cast(Stream[X], number_stream)
                else:
                    raise ValueError(f"Key must be provided if stream is not comprised of ints or floats.")
            else:
                return Stream(sorted(elements, key=key, reverse=not ascending), expected_size=len(elements))
        else:
            return Stream.empty()
    
    @overload
    def min(self:       "Stream[int]",
            default:    Y,
            key:        Callable[[X|Y],int|float] = ...) -> X|Y: ...
    
    @overload
    def min(self:       "Stream[float]",
            default:    Y,
            key:        Callable[[X|Y],int|float] = ...) -> X|Y: ...
    
    @overload
    def min(self,
            default:    Y,
            key:        Callable[[X|Y],int|float]) -> X|Y: ...
    
    def min(self, 
            default:    Y,
            key:        Callable[[X|Y],int|float]|None = None) -> X|Y:
        
        return min(self, key=cast(Callable[[X|Y],int|float], key), default=default)
    
    @overload
    def max(self:       "Stream[int]",
            default:    Y,
            key:        Callable[[X],int|float] = ...) -> X|Y: ...
    
    @overload
    def max(self:       "Stream[float]",
            default:    Y,
            key:        Callable[[X],int|float] = ...) -> X|Y: ...
    
    @overload
    def max(self,
            default:    Y,
            key:        Callable[[X],int|float]) -> X|Y: ...
    
    def max(self, 
            default:    Y,
            key:        Callable[[X],int|float]|None = None) -> X|Y:
        
        return max(self, key=cast(Callable[[X|Y],int|float], key), default=default)
    
    @overload
    def min_max(self:       "Stream[int]",
                default:    Y,
                key:        Callable[[X],int|float] = ...) -> Tuple[X|Y,X|Y]: ...
    
    @overload
    def min_max(self:       "Stream[float]",
                default:    Y,
                key:        Callable[[X],int|float] = ...) -> Tuple[X|Y,X|Y]: ...
    
    @overload
    def min_max(self,
                default:    Y,
                key:        Callable[[X],int|float]) -> Tuple[X|Y,X|Y]: ...
    
    def min_max(self, 
                default:    Y,
                key:        Callable[[X],int|float]|None = None) -> Tuple[X|Y,X|Y]:
        
        iterator = iter(self)
        try:
            x = next(iterator)
            min_x, max_x = x, x
        except StopIteration:
            return default, default
        
        for x in iterator:
            min_x = min(x, min_x, key=cast(Callable[[X|Y],int|float], key))
            max_x = max(x, max_x, key=cast(Callable[[X|Y],int|float], key))

        return min_x, max_x
    
    def scan(self, start: Y, scanner: Callable[[Y,X],Y]) -> "Stream[Y]":
        def iterator() -> Iterator[Y]:   
            y = start
            for x in self:
                y = scanner(y,x)
                yield y

        return Stream(iterator(), expected_size=self._expected_length)
    
    def peek(self, f: Callable[[X],None]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                f(x)
                yield x

        return Stream(iterator(), expected_size=self._expected_length)
    
    def foreach(self, f: Callable[[X],None]) -> None:
        for x in self:
            f(x)

    def chain(self, other: Iterable[X]) -> "Stream[X]":
        def iterator() -> Iterator[X]:
            for x in self:
                yield x

            for x in other:
                yield x

        if self._expected_length is not None:
            if isinstance(other, Sized):
                expected_length = self._expected_length + len(other)
            elif isinstance(other, Stream) and other._expected_length is not None:
                expected_length = self._expected_length + other._expected_length
            else:
                expected_length = None
        else:
            expected_length = None

        return Stream(iterator(), expected_size=expected_length)
    
    @overload
    def group_by(self:      "Stream[Tuple[Y,Z]]",
                 *,
                 key:       Callable[[X],Tuple[Y,Z]] = ...,
                 keep_key:  Literal[False] = ...) -> "Stream[List[Z]]": ...

    @overload
    def group_by(self, 
                 *,
                 key:       Callable[[X],Tuple[Y,Z]],
                 keep_key:  Literal[False] = ...) -> "Stream[List[Z]]": ...

    @overload
    def group_by(self:      "Stream[Tuple[Y,Z]]",
                 *,
                 key:       Callable[[X],Tuple[Y,Z]] = ...,
                 reduce:    Callable[[Z,Z],Z],
                 keep_key:  Literal[False] = ...) -> "Stream[Z]": ...

    @overload
    def group_by(self, 
                 *,
                 key:       Callable[[X],Tuple[Y,Z]], 
                 reduce:    Callable[[Z,Z],Z],
                 keep_key:  Literal[False] = ...) -> "Stream[Z]": ...
    
    @overload
    def group_by(self:      "Stream[Tuple[Y,Z]]",
                 *,
                 key:       Callable[[X],Tuple[Y,Z]] = ...,
                 keep_key:  Literal[True]) -> "Stream[Tuple[Y,List[Z]]]": ...

    @overload
    def group_by(self, 
                 *,
                 key:       Callable[[X],Tuple[Y,Z]],
                 keep_key:  Literal[True]) -> "Stream[Tuple[Y,List[Z]]]": ...

    @overload
    def group_by(self:      "Stream[Tuple[Y,Z]]",
                 *,
                 key:       Callable[[X],Tuple[Y,Z]] = ...,
                 reduce:    Callable[[Z,Z],Z],
                 keep_key:  Literal[True]) -> "Stream[Tuple[Y,Z]]": ...

    @overload
    def group_by(self, 
                 *,
                 key:       Callable[[X],Tuple[Y,Z]], 
                 reduce:    Callable[[Z,Z],Z],
                 keep_key:  Literal[True]) -> "Stream[Tuple[Y,Z]]": ...
        
    def group_by(self, 
                 *,
                 key:       Callable[[X],Tuple[Y,Z]]|None = None,
                 reduce:    Callable[[Z,Z],Z]|None = None,
                 keep_key:  bool = False) -> "Stream[Tuple[Y,List[Z]|Z]|List[Z]|Z]":
        
        stream = Stream(self.dict(key=key, reduce=reduce).items()) # type: ignore

        if keep_key:
            return stream
        else: 
            return stream.map(lambda key_value: key_value[1])

    def all(self, f: Callable[[X],bool]) -> bool:
        return all(f(x) for x in self)
    
    def any(self, f: Callable[[X],bool]) -> bool:
        return any(f(x) for x in self)
    
    def fork(self,
             type:          Literal["process","thread"],
             f:             Callable[[X],Y],
             max_forks:     int|None = None) -> "Stream[Y]":
        if type == "process":
            return self.fork_processes(f, max_forks)
        elif type == "thread":
            return self.fork_threads(f, max_forks)
        else:
            assert_never(type)
    
    def fork_processes(self, 
                       f:             Callable[[X],Y], 
                       max_processes: int|None = None) -> "Stream[Y]":

        def iterator() -> Iterator[Y]:
            with Pool(processes=max_processes) as pool:
                for y in pool.imap(func=_DillPickle(f), iterable=self):
                    yield y

        return Stream(iterator(), expected_size=self._expected_length)
    
    def fork_threads(self,
                     f:             Callable[[X],Y],
                     max_threads:   int|None = None) -> "Stream[Y]":
        def iterator() -> Iterator[Y]:
            with ThreadPool(processes=max_threads) as threads:
                for y in threads.imap(func=f, iterable=self):
                    yield y

        return Stream(iterator(), expected_size=self._expected_length)

    def monitor(self, 
                description:        str|None = None,
                expected_length:    int|None = None) -> "Stream[X]":
        if expected_length is None:
            expected_length = self._expected_length

        def iterator() -> Iterator[X]:
            for x in tqdm(iterable=self, desc=description, total=expected_length):
                yield x

        if self._expected_length is None:
            return Stream(iterator(), expected_size=expected_length)
        else:
            return Stream(iterator(), expected_size=self._expected_length)
    
    def item(self) -> X:
        iterator = iter(self)
        X = next(iterator)
        try:
            X2 = next(iterator)
        except StopIteration:
            return X
        
        raise ValueError(f"Stream must be singular, but contained more elements: {X=}, {X2=}")
    
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
                        raise TypeError(f"Implicit dict construction failed. Expected Stream[Tuple[Y,Z]] if key is not provided, but stream element type was: {type(x)}")
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
                        raise TypeError(f"Implicit dict construction failed. Expected Stream[Tuple[Y,Z]] if key is not provided, but stream element type was: {type(x)}")
            else:
                for x in self:
                    y,z2 = key(x)
                    z1 = key_z.get(y)
                    if z1 is None:
                        key_z[y] = z2
                    else:
                        key_z[y] = reduce(z1, z2)

            return key_z
        
    def ref_list(self, 
                 eviction_policy:   "EvictionPolicy",
                 location:          "Location",
                 max_memory:        Memory|None = None,
                 max_entries:       int|None = None, 
                 verbose:           bool = False) -> "RefList[X]":
        from .reflist import RefList
        return RefList(
            entries=self,
            eviction_policy=eviction_policy,
            location=location,
            max_memory=max_memory,
            max_entries=max_entries,
            verbose=verbose
        )
    
    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            for x in self:
                dill.dump(x, file)

    @staticmethod
    def load(path:              str, 
             obj_type:          Type[Y], 
             *,
             strict_type_check: bool = True,
             raise_type_error:  bool = True) -> "Stream[Y]":
        def iterator() -> Iterator[Y]:
            y_type = cast(Type[Y], obj_type.mro()[0])
            with open(path, "rb") as file:
                while True:
                    try:
                        obj = dill.load(file)
                        if strict_type_check:
                            if type(obj) is y_type:
                                yield obj
                            elif raise_type_error:
                                raise TypeError(f"Unpickled object is of incorrect type: {type(obj)}, expected: {y_type}")
                        else:
                            if isinstance(obj, y_type):
                                yield obj
                            elif raise_type_error:
                                raise TypeError(f"Unpickled object is of incorrect type: {type(obj)}, expected: {y_type}")
                    except EOFError:
                        break

        return Stream(iterator())
    
    @staticmethod
    def empty() -> "Stream[Never]":
        return Stream(tuple(), expected_size=0)
    
    @staticmethod
    def sample(population:          Sequence[Y],
               with_replacement:    bool,
               weights:             Sequence[int|float]|Callable[[Y],int|float]|None = None) -> "Stream[Y]":
        
        population_list = list(population)

        if not population_list:
            return Stream.empty()
        
        if isinstance(weights, Sequence):
            weights_list = list(weights)
            
            if len(weights_list) != len(population_list):
                raise ValueError(f"Length of population: {len(population_list)} differs from length of weights: {len(weights_list)}")
        else:
            weights_list = (Stream(population_list)
                    .map((lambda _: 1) if weights is None else weights)
                    .map(abs)
                    .list())
        
        if with_replacement:
            cum_weights = (Stream(weights_list)
                           .scan(0.0, lambda y,x: y+x)
                           .list())

            return Stream(lambda: random.choices(
                population=population_list,
                cum_weights=cum_weights,
                k=1
                )[0])
        else:
            def iterator() -> Iterator[Y]:
                while population_list:
                    try:
                        choice = random.choices(
                            population=range(len(population_list)),
                            weights=weights_list,
                            k=1
                            )[0]
                    except ValueError:
                        choice = random.choices(
                            population=range(len(population_list)),
                            k=1
                            )[0]
                    
                    weights_list.pop(choice)
                    yield population_list.pop(choice)

            return Stream(iterator(), expected_size=len(population_list))

    def run(self) -> None:
        for _ in self:
            pass
    
    def __add__(self, other: Iterable[X]) -> "Stream[X]":
        return self.chain(other)

    def __iter__(self) -> Iterator[X]:
        return iter(self._source)