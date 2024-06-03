from typing import Any

from xai import FeedForward, Lazy
from xai.network import FeedForward, Ints
from . import *
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Module

X = TypeVar("X", bound=Tuple[int,...])
L = TypeVar("L", bound=Tuple[int,...]) 

@dataclass
class AutoEncoderFeedForward(Generic[X,L], FeedForward[X,X]):
    parent:         "AutoEncoder[X,L]"
    input:          Lazy[Tensor]
    output:         Lazy[Tensor]
    embedding:      FeedForward[X,L]
    reconstruction: FeedForward[L,X]

    def explain_encoding(self, 
                         algorithm:     Explainer|Explainers, 
                         background:    Array|None) -> Explanation[L,X]:
        return self.embedding.explain(algorithm, background)
    
    def explain_reconstruction(self, 
                               algorithm:   Explainer|Explainers, 
                               background:  Array|None) -> Explanation[X,L]:
        return self.reconstruction.explain(algorithm, background)

class AutoEncoder(Generic[X,L], Network[X,X]):

    def __init__(self, 
                 data_shape:        X, 
                 latent_shape:      L,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 output_activation: Activation|None = None,
                 device:            Device = "auto") -> None:

        latent_shape = latent_shape
        explainers: Dict[Type[Explainer],Tuple[Array,Explainer]] = {}
        
        encoder = Network.dense(
            input_dim=data_shape,
            output_dim=latent_shape,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            device=device
        )
        decoder = Network.dense(
            input_dim=encoder.output_shape,
            output_dim=encoder.input_shape,
            hidden_layers=hidden_layers[::-1] if isinstance(hidden_layers, Sequence) else hidden_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            device=device
        )       

        super().__init__(
            device=device,
            input_shape=data_shape,
            output_shape=data_shape,
            logits=encoder + decoder
        )
        
        self.latent_shape = latent_shape
        self.explainers: Dict[Type[Explainer],Tuple[Array,Explainer]] = explainers
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X: Array|Lazy[Array]|FeedForward[Ints,X]) -> AutoEncoderFeedForward[X,L]:
        embedding = self.encoder(X)
        reconstruction = self.decoder(embedding.output)
        return AutoEncoderFeedForward(
            parent=self,
            input=embedding.input,
            output=reconstruction.output,
            embedding=embedding,
            reconstruction=reconstruction,
        )
    
    def __call__(self, X: Array|Lazy[Array]|FeedForward[Ints,X]) -> AutoEncoderFeedForward[X,L]:
        return super().__call__(X)