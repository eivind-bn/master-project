from . import *

import time as system_time

class Time(Unit["Time"]):

    def nanoseconds(self) -> "NanoSeconds":
        return NanoSeconds.of(self)

    def microseconds(self) -> "MicroSeconds":
        return MicroSeconds.of(self)

    def milliseconds(self) -> "MilliSeconds":
        return MilliSeconds.of(self)

    def seconds(self) -> "Seconds":
        return Seconds.of(self)
    
    def minutes(self) -> "Minutes":
        return Minutes.of(self)
    
    def hours(self) -> "Hours":
        return Hours.of(self)
    
    def days(self) -> "Days":
        return Days.of(self)
    
    @classmethod
    def now(cls: Type[Self]) -> Self:
        return cls.of(Seconds(system_time.time()))
    
    def __eq__(self, operand: object) -> bool:
        if isinstance(operand, Time):
            return self._value == self._convert_unit(operand)
        else:
            return False

class NanoSeconds(Time):
    order = Nano.order 

class MicroSeconds(Time):
    order = Micro.order

class MilliSeconds(Time):
    order = Milli.order

class Seconds(Time):
    pass

class Minutes(Time):
    order = Seconds.order*60

class Hours(Time):
    order = Minutes.order*60

class Days(Time):
    order = Hours.order*24