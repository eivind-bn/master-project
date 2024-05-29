from typing import Any
from . import *
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Module

Ints: TypeAlias = Tuple[int,...]
X = TypeVar("X", bound=Tuple[int,...])
L = TypeVar("L", bound=Tuple[int,...]) 

@dataclass
class AutoEncoderFeedForward(Generic[X,L]):
    parent:         "AutoEncoder[X,L]"
    input:          Tensor
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

class AutoEncoder(Generic[X,L], Serializable["AutoEncoder"]):

    def __init__(self, 
                 data_shape:        X, 
                 latent_shape:      L,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 output_activation: Activation|None = None,
                 device:            Device = "auto") -> None:
        
        self.latent_shape = latent_shape
        self.explainers: Dict[Type[Explainer],Tuple[Array,Explainer]] = {}
        
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
            hidden_layers=hidden_layers[::-1] if isinstance(hidden_layers, Sequence) else hidden_layers,
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
    def input_shape(self) -> X:
        return self.encoder.input_shape
    
    @property
    def output_shape(self) -> X:
        return self.decoder.output_shape

    def __call__(self, X: Array|Lazy[Array]) -> AutoEncoderFeedForward[X,X]:
        embedding = self.encoder(X)
        reconstruction = self.decoder(embedding.output())
        return AutoEncoderFeedForward(
            parent=self,
            input=embedding.input,
            embedding=embedding,
            reconstruction=reconstruction,
        )