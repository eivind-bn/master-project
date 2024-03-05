from typing import *
from dataclasses import dataclass
from torch import Tensor
from numpy.typing import NDArray

from .network import Network, Ints
from .activation import Activation
from . import Device

Sx = TypeVar("Sx", bound=Tuple[int,...])
Sy = TypeVar("Sy", bound=Tuple[int,...]) 

class AutoEncoder(Network[Sx,Sy]):
    @dataclass
    class FeedForward(Network.FeedForward):
        _embedding:    Callable[[], Tensor]|Tensor

        @property
        def embedding(self) -> Tensor:
            if not isinstance(self._embedding, Tensor):
                self._embedding = self._embedding()

            return self._embedding

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

    def __call__(self, array: NDArray[Any]|Tensor) -> FeedForward:
        input = self._to_tensor(array)
        encoding = self.encoder(input)
        decoding = self.decoder(encoding.output)
        return self.FeedForward(
            _input=input,
            _embedding=lambda: encoding.output,
            _output=lambda: decoding.output,
        )