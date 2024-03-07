from typing import *
from dataclasses import dataclass
from torch import Tensor
from numpy.typing import NDArray

from .network import Network, Ints, Array
from .activation import Activation
from . import Device
from .lazy import Lazy

Sx = TypeVar("Sx", bound=Tuple[int,...])
Sy = TypeVar("Sy", bound=Tuple[int,...]) 

class AutoEncoder(Generic[Sx,Sy], Network[Sx,Sx]):
    @dataclass
    class FeedForward(Network.FeedForward):
        embedding:      Lazy[Tensor]
        reconstruction: Lazy[Tensor]

    def __init__(self, 
                 data_shape:        Sx, 
                 latent_shape:      Sy,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 output_activation: Activation|None = None,
                 device:            Device = "auto") -> None:
        
        self.latent_shape = latent_shape
        
        self.encoder = Network.dense(
            input_dim=data_shape,
            output_dim=latent_shape,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            device=device
        )

        self.decoder = Network.dense(
            input_dim=self.encoder.output_shape,
            output_dim=self.encoder.input_shape,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            device=device
        )

    @property
    def logits(self) -> Tuple[Callable[[Tensor],Tensor],...]:
        return (self.encoder + self.decoder).logits
            
    @property
    def device(self) -> Device:
        return self.encoder.device
   
    @property
    def input_shape(self) -> Sx:
        return self.encoder.input_shape
    
    @property
    def output_shape(self) -> Sx:
        return self.decoder.output_shape

    def __call__(self, array: Array|Lazy[Array]) -> FeedForward:
        input = Lazy(lambda: array).map(self._to_tensor)
        encoding = self.encoder(input)
        decoding = self.decoder(encoding.output)
        return self.FeedForward(
            input=input,
            embedding=Lazy(encoding.output),
            reconstruction=Lazy(decoding.output),
            output=Lazy(decoding.output),
        )