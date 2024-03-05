from typing import *
from dataclasses import dataclass
from .network import Network
from .reflist import RefList
from .bytes import GigaBytes
from .explainer import Explainer
from .stats import TrainStats
from .feed_forward import FeedForward
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, uint8
from . import Device, get_device
from .activation import Activation
from .autoencoder import AutoEncoder

import mnist # type: ignore
import numpy as np
import torch

Sx: TypeAlias = Tuple[Literal[28],Literal[28]]
Sy: TypeAlias = Tuple[Literal[10]]
Sl = TypeVar("Sl", bound=Tuple[int,...])

class MNIST(Generic[Sl], Network[Sx,Sy]):
    @dataclass
    class FeedForward(Network.FeedForward):
        _embedding: Callable[[],Tensor]|Tensor
        _decode: Callable[[],Tensor]|Tensor

        @property
        def decode(self) -> Tensor:
            if not isinstance(self._decode, Tensor):
                self._decode = self._decode()

            return self._decode
        
        @property
        def embedding(self) -> Tensor:
            if not isinstance(self._embedding, Tensor):
                self._embedding = self._embedding()

            return self._embedding
        
        def digit(self) -> Tuple[int,...]:
            return tuple(self.output.argmax(dim=1))

    def __init__(self, 
                 latent_shape:      Sl,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 output_activation: Activation|None = None,
                 device:            Device = "auto") -> None:
        super().__init__(
            input_shape=(28,28),
            output_shape=(10,),
            device=device
        )

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

    def __call__(self, array: NDArray[Any]|Tensor) -> FeedForward:
        input = self._to_tensor(array)
        auto_code = self.autoencoder(array)
        classification = self.classifier_head(auto_code.embedding)
        return MNIST.FeedForward(
            _input=input,
            _output=lambda: classification.output,
            _embedding=lambda: auto_code.embedding,
            _decode=lambda: auto_code.output
        )

    