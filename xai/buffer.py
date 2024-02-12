from typing import *
from collections import deque
from typing import Iterator
from tqdm import tqdm

from .bytes import Memory, Bytes
from .cache import Cache, Dump, Load
from .stream import Stream

import random
import itertools
import copy

T = TypeVar("T")

EvictionPolicy = Literal["FIFO", "Random", "Reject", "Throw"]

class AppendReject(Exception):
    pass

class Buffer(Stream[T], Generic[T]):

    def __init__(self, 
                 entries:           Iterable[Cache[T]|T],
                 eviction_policy:   EvictionPolicy,
                 use_ram:           bool,
                 max_memory:        Memory|None,
                 max_entries:       int|None = None, 
                 verbose:           bool = False) -> None:
        super().__init__(self)
        
        match eviction_policy:
            case "FIFO":
                def evict() -> Cache[T]:
                    return self._entries.pop(0)
            case "Random":
                def evict() -> Cache[T]:
                    idx = random.randrange(0, len(self._entries))
                    cache = self._entries.pop(idx)
                    return cache
            case "Reject":
                def evict() -> Cache[T]:
                    raise AppendReject()
            case "Throw":
                def evict() -> Cache[T]:
                    raise OverflowError(f"Buffer is full.")
            case _:
                assert_never(eviction_policy)

        self._evict = evict
        self._eviction_policy = eviction_policy
        self._use_ram = use_ram
        self._max_memory = max_memory
        self._max_entries = max_entries
        self._entries: List[Cache[T]] = []

        self.extend(entries, verbose=verbose)

    @property
    def max_memory(self) -> Memory|None:
        return self._max_memory

    def byte_size(self) -> Bytes:
        return sum((entry.size() for entry in self._entries), start=Bytes(0))
    
    def entry_size(self) -> int:
        return len(self._entries)
    
    def randoms(self, with_replacement: bool) -> Stream[T]:
        def iterator() -> Iterator[T]:
            N = len(self._entries)
            if with_replacement:
                while True:
                    idx = random.randrange(0,N)
                    with self._entries[idx] as data:
                        yield data
            else:
                indices = list(range(0,N))
                while indices:
                    idx = random.randrange(0,len(indices))
                    with self._entries[indices.pop(idx)] as data:
                        yield data

        return Stream(iterator())
    
    def extend(self, entries: Iterable[Cache[T]|T], verbose: bool = False) -> None:
        byte_size = self.byte_size()
        entry_size = self.entry_size()

        def get_desc() -> str:
            loc = "RAM" if self._use_ram else "Disk"
            used_gigs = byte_size.gigabytes().float()
            if self._max_memory is not None:
                max_gigs = self._max_memory.gigabytes().float()
                return f"{loc} used: {used_gigs:.2f}/{max_gigs:.2f}GB"
            else:
                return f"{loc} used: {used_gigs:.2f}GB"

        try:
            with tqdm(disable=not verbose) as bar:
                for entry in entries:
                    if isinstance(entry, Cache):
                        entry = entry.loaded() if self._use_ram else entry.dumped()
                    else:
                        entry = Load(entry) if self._use_ram else Dump(entry)

                    new_size = byte_size + entry.size()
                    if self._max_memory is not None:
                        if new_size > self._max_memory:
                            while new_size > self._max_memory:
                                new_size -= self._evict().size()

                    if self._max_entries is not None:
                        while not entry_size < self._max_entries:
                            self._evict()
                            entry_size -= 1

                    self._entries.append(entry)
                    byte_size = new_size

                    bar.set_description(get_desc())
                    bar.update()
        except AppendReject:
            pass

    def extended(self, other: Iterable[Cache[T]|T], verbose: bool = False) -> "Buffer[T]":
        buffer = self.copy()
        buffer.extend(other, verbose)
        return buffer

    def pop(self, loc: int) -> T:
        with self._entries.pop(loc) as data:
            return data

    def pop_range(self, end: int, start: int = 0, step: int = 1) -> Stream[T]:       
        def loader() -> Iterator[T]:
            for i in range(start, end, step):
                with self._entries.pop(start) as data:
                    yield data

        return Stream(loader())
    
    def remove_range(self, end: int, start: int = 0, step: int = 1) -> None:
        for i in range(start, end, step):
            self._entries.pop(i)
        
    def copy(self) -> Self:
        buffer = copy.copy(self)
        buffer._entries = self._entries.copy()
        return buffer

    def __add__(self, other: Iterable[Cache[T]|T]) -> "Buffer[T]":
        return self.extended(other)

    def __getitem__(self, loc: int|slice|Iterable[int]) -> Stream[T]:
        def load() -> Iterator[T]:
            if isinstance(loc, slice):
                entries: Iterable[Cache[T]] = self._entries[loc]
            elif isinstance(loc, Iterable):
                entries = iter(self._entries[i] for i in loc)
            else:
                entries = self._entries[loc:loc+1]

            for entry in entries:
                with entry as data:
                    yield data

        return Stream(load())
        
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        use_ram = self._use_ram
        entries = len(self._entries)
        size = self.byte_size().megabytes().float()
        if self._max_memory:
            capacity = (size / self._max_memory.megabytes().float())*100
            return f"{cls_name}({use_ram=}, {entries=}, {size=:.2f}MB, {capacity=:.2f}%)"
        return f"{cls_name}({use_ram=}, {entries=}, {size=:.2f}MB)"
    
    def __iter__(self) -> Iterator[T]:
        for entry in self._entries:
            with entry as data:
                yield data

