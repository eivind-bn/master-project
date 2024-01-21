from typing import *
from .si import *

import psutil

# Metric system

class Memory(Unit["Memory"]):

    def bytes(self) -> "Bytes":
        return Bytes.of(self)

    def kilobytes(self) -> "KiloBytes":
        return KiloBytes.of(self)

    def megabytes(self) -> "MegaBytes":
        return MegaBytes.of(self)

    def gigabytes(self) -> "GigaBytes":
        return GigaBytes.of(self)

    def terabytes(self) -> "TeraBytes":
        return TeraBytes.of(self)

    def petabytes(self) -> "PetaBytes":
        return PetaBytes.of(self)

    def exabytes(self) -> "ExaBytes":
        return ExaBytes.of(self)

    def kibibytes(self) -> "KibiBytes":
        return KibiBytes.of(self)

    def mebibytes(self) -> "MebiBytes":
        return MebiBytes.of(self)

    def gibibytes(self) -> "GibiBytes":
        return GibiBytes.of(self)

    def tebibytes(self) -> "TebiBytes":
        return TebiBytes.of(self)

    def pebibytes(self) -> "PebiBytes":
        return PebiBytes.of(self)

    def exbibytes(self) -> "ExbiBytes":
        return ExbiBytes.of(self)

    def zebibytes(self) -> "ZebiBytes":
        return ZebiBytes.of(self)

    def yobibytes(self) -> "YobiBytes":
        return YobiBytes.of(self)
    
    @classmethod
    def ram_total(cls: Type[Self]) -> Self:
        return cls.of(Bytes(psutil.virtual_memory().total))
    
    @classmethod
    def ram_used(cls: Type[Self]) -> Self:
        return cls.of(Bytes(psutil.virtual_memory().used))

    @classmethod
    def ram_free(cls: Type[Self]) -> Self:
        return cls.of(Bytes(psutil.virtual_memory().free))

    @classmethod
    def ram_available(cls: Type[Self]) -> Self:
        return cls.of(Bytes(psutil.virtual_memory().available))

    def __eq__(self, operand: object) -> bool:
        if isinstance(operand, Bytes):
            return self._value == self._convert_unit(operand)
        else:
            return False

class Bytes(Memory):
    pass

class KiloBytes(Memory):
    order = Kilo.order

class MegaBytes(Memory):
    order = Mega.order

class GigaBytes(Memory):
    order = Giga.order

class TeraBytes(Memory):
    order = Tera.order

class PetaBytes(Memory):
    order = Peta.order

class ExaBytes(Memory):
    order = Exa.order


# Base 2 system

class KibiBytes(Memory):
    order = 1024

class MebiBytes(Memory):
    order = 1024**2

class GibiBytes(Memory):
    order = 1024**3

class TebiBytes(Memory):
    order = 1024**4

class PebiBytes(Memory):
    order = 1024**5

class ExbiBytes(Memory):
    order = 1024**6

class ZebiBytes(Memory):
    order = 1024**7

class YobiBytes(Memory):
    order = 1024**8
