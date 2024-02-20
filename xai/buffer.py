from typing import *
from collections import deque
from typing import Iterator
from tqdm import tqdm

from .bytes import Memory, Bytes
from .cache import Cache, Dump, Load
from .stream import Stream

import random
import copy

T = TypeVar("T")

EvictionPolicy = Literal["FIFO", "Random", "Reject", "Throw"]

class EntryRejection(Exception):
    pass

class Buffer(Stream[T], Generic[T]):

    def __init__(self, 
                 entries:           Iterable[Cache[T]|T],
                 eviction_policy:   EvictionPolicy,
                 use_ram:           bool,
                 max_memory:        Memory|None = None,
                 max_entries:       int|None = None, 
                 verbose:           bool = False) -> None:
        super().__init__(self)

        def fifo_evict() -> Cache[T]:
            return self._entries.pop(0)
        
        def random_evict() -> Cache[T]:
            idx = random.randrange(0, len(self._entries))
            cache = self._entries.pop(idx)
            return cache
        
        def reject_evict() -> Cache[T]:
            raise EntryRejection()
        
        def throw_evict() -> Cache[T]:
            raise OverflowError(f"Buffer is full.")

        self._eviction_policies: Dict[EvictionPolicy,Callable[[],Cache[T]]] = {
            "FIFO": fifo_evict,
            "Random": random_evict,
            "Reject": reject_evict,
            "Throw": throw_evict
        }

        self._eviction_policy = eviction_policy
        self._evict = self._eviction_policies[eviction_policy]
        self._use_ram = use_ram
        self._max_memory = max_memory
        self._max_entries = max_entries
        self._verbose = verbose
        self._byte_size: Memory|None = None
        self._entries: List[Cache[T]] = []

        self.extend(entries, verbose=verbose)

    @property
    def use_ram(self) -> bool:
        return self._use_ram

    @property
    def max_memory(self) -> Memory|None:
        return self._max_memory
    
    @property
    def max_entries(self) -> int|None:
        return self._max_entries

    def byte_size(self) -> Bytes:
        if self._byte_size is None:
            self._byte_size = sum((entry.size() for entry in self._entries), start=Bytes(0))
        return self._byte_size
    
    def entry_size(self) -> int:
        return len(self._entries)
    
    def randoms(self, with_replacement: bool) -> Stream[T]:
        def iterator() -> Iterator[T]:
            N = self.entry_size()
            if with_replacement:
                while True:
                    idx = random.randrange(0,N)
                    with self._entries[idx] as data:
                        self._byte_size = None
                        yield data
            else:
                indices = list(range(0,N))
                while indices:
                    idx = random.randrange(0,len(indices))
                    with self._entries[indices.pop(idx)] as data:
                        self._byte_size = None
                        yield data

        return Stream(iterator())
    
    def append(self,
               entry:           Cache[T]|T,
               eviction_policy: EvictionPolicy|None = None) -> None:
        self.extend(
            entries=(entry, ),
            eviction_policy=eviction_policy
        )
    
    def extend(self, 
               entries:         Iterable[Cache[T]|T], 
               eviction_policy: EvictionPolicy|None = None,
               verbose:         bool = False) -> None:
        byte_size = self.byte_size()
        entry_size = self.entry_size()

        if eviction_policy is None:
            evict = self._evict
        else:
            evict = self._eviction_policies[eviction_policy]

        try:
            with tqdm(disable=not verbose) as bar:
                for entry in entries:
                    entry = self._to_cache(entry)

                    new_size = byte_size + entry.size()
                    if self._max_memory is not None:
                        if new_size > self._max_memory:
                            while new_size > self._max_memory:
                                new_size -= evict().size()

                    if self._max_entries is not None:
                        while not entry_size < self._max_entries:
                            evict()
                            entry_size -= 1

                    self._entries.append(entry)
                    byte_size = new_size

                    bar.set_description(self._get_load_text(byte_size))
                    bar.update()
        except EntryRejection:
            pass

        self._byte_size = byte_size

    def appended(self,
                 other:             Cache[T]|T, 
                 eviction_policy:   EvictionPolicy|None = None) -> "Buffer[T]":
        return self.extended(
            other=(other,),
            eviction_policy=eviction_policy
        )

    def extended(self, 
                 other:             Iterable[Cache[T]|T], 
                 eviction_policy:   EvictionPolicy|None = None,
                 verbose:           bool = False) -> "Buffer[T]":
        buffer = self.copy()
        buffer.extend(
            entries=other,
            eviction_policy=eviction_policy,
            verbose=verbose
        )
        return buffer
    
    def new_like(self, entries: Iterable[Cache[T]|T]) -> "Buffer[T]":
        return Buffer(
            entries=entries,
            eviction_policy=self._eviction_policy,
            use_ram=self._use_ram,
            max_memory=self._max_memory,
            max_entries=self._max_entries,
            verbose=self._verbose
        )
    
    @overload
    def pop(self, loc: int) -> T: ...

    @overload
    def pop(self, loc: Sequence[int]) -> Stream[T]: ...

    def pop(self, loc: int|Sequence[int]) -> T|Stream[T]:
        self._byte_size = None
        if isinstance(loc, Sequence):
            keep_flags = [True]*self.entry_size()
            for index in loc:
                keep_flags[index] = False

            keep_list: List[Cache[T]] = []
            pop_list: List[Cache[T]] = []
            
            for i,keep in enumerate(keep_flags):
                if keep:
                    keep_list.append(self._entries[i])
                else:
                    pop_list.append(self._entries[i])

            def loader() -> Iterator[T]:
                for pop_item in pop_list:
                    with pop_item as data:
                        yield data

            self._entries = keep_list
            return Stream(loader())
        else:
            with self._entries[loc] as data:
                return data
            
    def remove(self, loc: int|Sequence[int]) -> None:
        self._byte_size = None
        self.pop(loc)

    def replace(self, 
                entries:            Iterable[Tuple[int,Cache[T]|T]],
                eviction_policy:    EvictionPolicy|None = None,
                verbose:            bool = False) -> None:

        byte_size = self.byte_size()

        if eviction_policy is None:
            evict = self._evict
        else:
            evict = self._eviction_policies[eviction_policy]
        
        try:
            with tqdm(disable=not verbose) as bar:
                for index,new_value in entries:
                    old_value = self._entries[index]
                    new_value = self._to_cache(new_value)

                    if self._max_memory is not None:
                        old_value_size = old_value.size()
                        new_value_size = new_value.size()
                        new_size = byte_size + new_value_size - old_value_size
                        while new_size > self._max_memory:
                            new_size -= evict().size()
                        
                    self._entries[index] = new_value
                    byte_size = new_size

                    bar.set_description(self._get_load_text(byte_size))
                    bar.update()
        except EntryRejection:
            pass

        self._byte_size = byte_size

    def copy(self) -> "Buffer[T]":
        return Buffer(
            entries=self._entries.copy(),
            eviction_policy=self._eviction_policy,
            use_ram=self.use_ram,
            max_memory=self.max_memory,
            max_entries=self.max_entries,
            verbose=self._verbose
        )
    
    def clear(self) -> None:
        self._byte_size = None
        self._entries.clear()

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
                    self._byte_size = None
                    yield data

        return Stream(load())

    def _to_cache(self, data: Cache[T]|T) -> Cache[T]:
        if isinstance(data, Cache):
            return data.loaded() if self._use_ram else data.dumped()
        else:
            return Load(data) if self._use_ram else Dump(data)
        
    def _get_load_text(self, current_size: Bytes) -> str:
        loc = "RAM" if self._use_ram else "Disk"
        used_gigs = current_size.gigabytes().float()
        if self._max_memory is not None:
            max_gigs = self._max_memory.gigabytes().float()
            return f"{loc} used: {used_gigs:.2f}/{max_gigs:.2f}GB"
        else:
            return f"{loc} used: {used_gigs:.2f}GB"
        
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
                self._byte_size = None
                yield data

