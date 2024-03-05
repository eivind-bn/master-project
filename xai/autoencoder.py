from typing import *

from .network import Network
from .activation import Activation
from . import Device

Sx = TypeVar("Sx", bound=Tuple[int,...])
Sy = TypeVar("Sy", bound=Tuple[int,...])

class AutoEncoder(Generic[Sx,Sy]):

    def __init__(self, 
                 data_shape:        Sx, 
                 latent_shape:      Sy,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 output_activation: Activation|None = None,
                 device:            Device = "auto") -> None:
        
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

        self.autoencoder = self.encoder + self.decoder

    def __call__(self, tensor: Tensor) -> Tensor:
        