from . import *
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Module

Sx = TypeVar("Sx", bound=Tuple[int,...])
Sy = TypeVar("Sy", bound=Tuple[int,...]) 

class AutoEncoder(Generic[Sx,Sy], Network[Sx,Sx]):
    @dataclass
    class FeedForward(Network.FeedForward):
        embedding:      Network.FeedForward
        reconstruction: Network.FeedForward

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

        super().__init__()

    @property
    def modules(self) -> Tuple[Module,...]:
        return (self.encoder + self.decoder).modules
            
    @property
    def device(self) -> Device:
        return self.encoder.device
   
    @property
    def input_shape(self) -> Sx:
        return self.encoder.input_shape
    
    @property
    def output_shape(self) -> Sx:
        return self.decoder.output_shape

    def __call__(self, X: Array|Lazy[Array]|"AutoEncoder.FeedForward") -> FeedForward:
        embedding = self.encoder(X)
        reconstruction = self.decoder(embedding)
        return self.FeedForward(
            parent=self,
            input=embedding.input,
            embedding=embedding,
            reconstruction=reconstruction,
            output=reconstruction.output,
        )