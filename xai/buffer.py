from typing import *
from collections import deque
from typing import Iterator
from tqdm import tqdm

from .bytes import Memory, Bytes
from .cache import Cache, Dump, Load
from .stream import Stream

import random

T = TypeVar("T")

EvictionPolicy = Literal["FIFO", "Random", "Reject"]

class Buffer(Generic[T], Sequence[Cache[T]], Stream[Cache[T]]):

    def __init__(self, 
                 entries:           Iterable[Cache[T]|T],
                 eviction_policy:   EvictionPolicy,
                 use_ram:           bool,
                 max_memory:        Memory|None,
                 max_entries:       int|None = None, 
                 verbose:           bool = False) -> None:
        self._eviction_policy = eviction_policy
        self._use_ram = use_ram
        self._max_memory = max_memory
        self._max_entries = max_entries

        used_memory = Bytes(0)
        data: Deque[Cache[T]] = deque(maxlen=max_entries)

        match self._eviction_policy:
            case "FIFO":
                def evict() -> Cache[T]:
                    return data.popleft()
            case "Random":
                def evict() -> Cache[T]:
                    idx = random.randrange(0, len(data))
                    cache = data[idx]
                    del data[idx]
                    return cache
            case "Reject":
                def evict() -> Cache[T]:
                    raise StopIteration
            case _:
                assert_never(eviction_policy)

        def append_with_capacity_check(cache: Cache[T]) -> None:
            nonlocal used_memory
            if self._max_memory is not None:
                new_size = used_memory + cache.size()
                if new_size < self._max_memory:
                    data.append(cache)
                else:
                    while new_size > self._max_memory:
                        new_size -= evict().size()

                    data.append(cache)

                used_memory = new_size

            else:
                data.append(cache)

        def get_desc() -> str:
            loc = "RAM" if use_ram else "Disk"
            used_gigs = used_memory.gigabytes().float()
            if max_memory is not None:
                max_gigs = max_memory.gigabytes().float()
                return f"{loc} used: {used_gigs:.2f}/{max_gigs:.2f}GB"
            else:
                return f"{loc} used: {used_gigs:.2f}GB"

        try:  
            if use_ram:
                with tqdm(disable=not verbose) as bar:
                    for entry in entries:
                        if isinstance(entry, Cache):
                            append_with_capacity_check(entry.loaded())
                        else:
                            append_with_capacity_check(Load(entry))
                            
                        bar.set_description(get_desc())
                        bar.update()
            else:
                with tqdm(disable=not verbose) as bar:
                    for entry in entries:
                        if isinstance(entry, Cache):
                            append_with_capacity_check(entry.dumped())
                        else:
                            append_with_capacity_check(Dump(entry))
                        
                        bar.set_description(get_desc())
                        bar.update()
        except StopIteration:
            pass

        self.entries: Tuple[Cache[T],...] = tuple(data)
        super().__init__(self)

    @property
    def max_memory(self) -> Memory|None:
        return self._max_memory

    def size(self) -> Bytes:
        return sum((entry.size() for entry in self.entries), start=Bytes(0))
    
    def length(self) -> int:
        return len(self.entries)
    
    def randoms(self, with_replacement: bool) -> Stream[Cache[T]]:
        def iterator() -> Iterator[Cache[T]]:
            N = len(self.entries)
            if with_replacement:
                while True:
                    idx = random.randrange(0,N)
                    yield self[idx]
            else:
                indices = list(range(0,N))
                while indices:
                    idx = random.randrange(0,len(indices))
                    yield self[indices.pop(idx)]

        return Stream(iterator())

    def appended(self, data: Cache[T]|T) -> "Buffer[T]":
        return self.extended((data,))
    
    def extended(self, other: Iterable[Cache[T]|T]) -> "Buffer[T]":
        return Buffer(
            eviction_policy=self._eviction_policy,
            entries=self.entries + tuple(other),
            max_entries=self._max_entries,
            max_memory=self._max_memory,
            use_ram=self._use_ram,
            )

    def __add__(self, other: Iterable[Cache[T]|T]) -> "Buffer[T]":
        return self.extended(other)
    
    @overload
    def __getitem__(self, loc: SupportsIndex) -> Cache[T]: ...

    @overload
    def __getitem__(self, loc: slice) -> Tuple[Cache[T],...]: ...

    def __getitem__(self, loc: SupportsIndex|slice) -> Cache[T]|Tuple[Cache[T],...]:
        if isinstance(loc, SupportsIndex):
            return self.entries[loc]
        else:
            def load() -> Iterator[Cache[T]]:
                for entry in self.entries[loc]:
                    yield entry

            return tuple(load())
        
    def __len__(self) -> int:
        return self.length()
        
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        use_ram = self._use_ram
        entries = len(self.entries)
        size = self.size().megabytes().float()
        if self._max_memory:
            capacity = (size / self._max_memory.megabytes().float())*100
            return f"{cls_name}({use_ram=}, {entries=}, {size=:.2f}MB, {capacity=:.2f}%)"
        return f"{cls_name}({use_ram=}, {entries=}, {size=:.2f}MB)"
    
    def __iter__(self) -> Iterator[Cache[T]]:
        for entry in self.entries:
            yield entry

