from typing import *
from dataclasses import dataclass
from .network import Network, Array
from .reflist import RefList
from .bytes import GigaBytes
from .stats import TrainStats
from .feed_forward import FeedForward
from .lazy import Lazy
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, uint8
from . import Device, get_device
from .activation import Activation
from .autoencoder import AutoEncoder
from .optimizer import SGD, Adam, RMSprop
from .loss import Loss

import mnist # type: ignore
import numpy as np
import torch

Sx: TypeAlias = Tuple[Literal[28],Literal[28]]
Sy: TypeAlias = Tuple[Literal[10]]
Sl = TypeVar("Sl", bound=Tuple[int,...])

class MNIST(Generic[Sl], Network[Sx,Sy]):
    @dataclass
    class FeedForward(Network.FeedForward):
        embedding:      Lazy[Tensor]
        reconstruction: Lazy[Tensor]
        classification: Lazy[Tensor]
        
        def digits(self) -> Tuple[int,...]:
            return tuple(self.classification().argmax(dim=1))

    def __init__(self, 
                 latent_shape:      Sl,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 output_activation: Activation|None = None,
                 device:            Device = "auto") -> None:

        images = torch.from_numpy(mnist.train_images()).to(device=device, dtype=torch.float32)
        images = images/255.0

        self.labels = torch.from_numpy(mnist.train_labels())
        labels_one_hot = torch.zeros((self.labels.shape[0],10)).float()
        labels_one_hot[torch.arange(0,self.labels.shape[0]),self.labels.int()] = 1.0

        self.autoencoder = AutoEncoder(
            data_shape=self.input_shape,
            latent_shape=latent_shape,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            device=device
        )

        self.classifier_head = Network.dense(
            input_dim=self.autoencoder.latent_shape,
            output_dim=(10,),
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            device=device
        )

        super().__init__(
            logits=(self.autoencoder.encoder + self.classifier_head).logits,
            input_shape=(28,28),
            output_shape=(10,),
            device=device
        )
    
    def fit_autoencoder(self, **params: Unpack[Adam.Params]) -> None:
        self.autoencoder.adam(**params).fit()
    
    def fit_classifier(self, 
                       epochs: int,
                       batch_size: int,
                       loss_criterion: Loss,
                       verbose: bool = False,
                       info: str | None = None,
                       **params: Unpack[Adam.Params]) -> None:
        self.classifier_head.adam(**params).fit()

    def __call__(self, array: Array|Lazy[Array]) -> FeedForward:
        input = Lazy(lambda: array).map(self._to_tensor)
        auto_code = self.autoencoder(array)
        classification = self.classifier_head(auto_code.embedding)
        return MNIST.FeedForward(
            input=input,
            output=Lazy(classification.output),
            embedding=Lazy(auto_code.embedding),
            reconstruction=Lazy(auto_code.output),
            classification=Lazy(classification.output)
        )

    