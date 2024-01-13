from typing import *

import psutil

class ByteSize:

    def __init__(self,
                 bytes: float = 0,
                 kilobytes: float = 0,
                 megabytes: float = 0,
                 gigabytes: float = 0,
                 terabytes: float = 0) -> None:
        self._byte_size = bytes +\
            (kilobytes * 1e3) +\
            (megabytes * 1e6) +\
            (gigabytes * 1e9) +\
            (terabytes * 1e12)
        
    def bytes(self) -> float:
        return self._byte_size
    
    def kilobytes(self) -> float:
        return self._byte_size * 1e-3
    
    def megabytes(self) -> float:
        return self._byte_size * 1e-6
    
    def gigabytes(self) -> float:
        return self._byte_size * 1e-9
    
    def terabytes(self) -> float:
        return self._byte_size * 1e-12
    
    def __add__(self, other: "ByteSize") -> "ByteSize":
        return ByteSize(
            bytes=self.bytes() + other.bytes()
        )
    
    def __sub__(self, other: "ByteSize") -> "ByteSize":
        return ByteSize(
            bytes=self.bytes() - other.bytes()
        )
    
    def __mul__(self, other: "ByteSize") -> "ByteSize":
        return ByteSize(
            bytes=self.bytes() * other.bytes()
        )
    
    def __div__(self, other: "ByteSize") -> "ByteSize":
        return ByteSize(
            bytes=self.bytes() / other.bytes()
        )
    
    def ram_occupancy(self) -> "RamOccupancy":
        return RamOccupancy(self)

class RamOccupancy:
    def __init__(self, bytes: ByteSize) -> None:
        self.byte_size = bytes

    def total(self) -> float:
        return self.byte_size.bytes() / ram_total().bytes()
    
    def used(self) -> float:
        return self.byte_size.bytes() / ram_used().bytes()
    
    def free(self) -> float:
        return self.byte_size.bytes() / ram_free().bytes()
    
    def available(self) -> float:
        return self.byte_size.bytes() / ram_available().bytes()
    
    
def ram_total() -> ByteSize:
    return ByteSize(
        bytes=psutil.virtual_memory().total
    )

def ram_used() -> ByteSize:
    return ByteSize(
        bytes=psutil.virtual_memory().used
    )

def ram_free() -> ByteSize:
    return ByteSize(
        bytes=psutil.virtual_memory().free
    )

def ram_available() -> ByteSize:
    return ByteSize(
        bytes=psutil.virtual_memory().available
    )