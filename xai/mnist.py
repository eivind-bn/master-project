from __future__ import annotations
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
from .explainer import Explainer, Explanation

import mnist # type: ignore
import numpy as np
import torch

Sx: TypeAlias = Tuple[Literal[28],Literal[28]]
Sy: TypeAlias = Tuple[Literal[10]]
Sl = TypeVar("Sl", bound=Tuple[int,...])

class MNIST(Generic[Sl], Network[Sx,Sy]):
    @dataclass
    class FeedForward(Network.FeedForward):
        _parent:        MNIST
        embedding:      Lazy[Tensor]
        reconstruction: Lazy[Tensor]
        classification: Lazy[Tensor]
        
        def digits(self) -> Tuple[int,...]:
            return tuple(self.classification().reshape((-1,10)).argmax(dim=1))
        
        def reconstruction_explanation(self, type: Type[Explainer], verbose: bool = False) -> Tuple[Explanation,...]:
            return self._parent.reconstruction_explainer(type).explain(self.embedding(), verbose=verbose)
        
        def classification_explanation(self, type: Type[Explainer], verbose: bool = False) -> Tuple[Explanation,...]:
            return self._parent.classifier_head_explainer(type).explain(self.embedding(), verbose=verbose)

    def __init__(self, 
                 latent_shape:      Sl,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 device:            Device = "auto") -> None:

        self._device = get_device(device)
        data = torch.from_numpy(mnist.train_images()).to(device=self.device, dtype=torch.float32)
        data = data/255.0
        indices = torch.randperm(len(data))
        train_portion = int(len(indices)*0.8)

        self.train_data = data[indices[:train_portion]]
        self.val_data = data[indices[train_portion:]]

        labels = torch.from_numpy(mnist.train_labels()).to(device=self.device, dtype=torch.long)
        self.train_labels = labels[indices[:train_portion]]
        self.val_labels = labels[indices[train_portion:]]

        self.autoencoder = AutoEncoder(
            data_shape=self.input_shape,
            latent_shape=latent_shape,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            output_activation="Sigmoid",
            device=device
        )
        self.autoencoder_optimizer = self.autoencoder.adam()

        self.classifier_head = Network.dense(
            input_dim=self.autoencoder.latent_shape,
            output_dim=self.output_shape,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            output_activation="Softmax",
            device=device
        )
        self.classifier_head_optimizer = self.classifier_head.adam()

        self._reconstruction_explainers: Dict[Type[Explainer],Explainer] = {}
        self._classifier_head_explainer: Dict[Type[Explainer],Explainer] = {}

    @property
    def logits(self) -> Tuple[Callable[[Tensor],Tensor],...]:
        return (self.autoencoder.encoder + self.classifier_head).logits
            
    @property
    def device(self) -> Device:
        return self._device
   
    @property
    def input_shape(self) -> Sx:
        return (28,28)
    
    @property
    def output_shape(self) -> Sy:
        return (10,)
    
    def reconstruction_explainer(self, type: Type[Explainer]) -> Explainer:
        explainer = self._reconstruction_explainers.get(type)
        if explainer is None:
            explainer = type(self.autoencoder.decoder, self(self.val_data).embedding())
            self._reconstruction_explainers[type] = explainer

        return explainer
    
    def classifier_head_explainer(self, type: Type[Explainer]) -> Explainer:
        explainer = self._classifier_head_explainer.get(type)
        if explainer is None:
            explainer = type(self.classifier_head, self(self.val_data).embedding())
            self._classifier_head_explainer[type] = explainer

        return explainer
    
    def fit_autoencoder(self, 
                        epochs: int,
                        batch_size: int,
                        loss_criterion: Loss,
                        verbose: bool = False,
                        info: str | None = None) -> TrainStats:
          self._reconstruction_explainers.clear()
          return self.autoencoder_optimizer.fit(
             X=self.train_data,
             Y=self.train_data,
             epochs=epochs,
             batch_size=batch_size,
             loss_criterion=loss_criterion,
             verbose=verbose,
             info=info
         )
    
    def fit_classifier(self, 
                       epochs: int,
                       batch_size: int,
                       loss_criterion: Loss,
                       verbose: bool = False,
                       info: str | None = None) -> TrainStats:
         return self.classifier_head_optimizer.fit(
             X=self(self.train_data).embedding(),
             Y=torch.nn.functional.one_hot(self.train_labels, num_classes=10).float(),
             epochs=epochs,
             batch_size=batch_size,
             loss_criterion=loss_criterion,
             verbose=verbose,
             info=info
         )

    def __call__(self, X: Array|Lazy[Array]) -> MNIST.FeedForward:
        autoencoder = self.autoencoder(X)
        classification = self.classifier_head(autoencoder.embedding)
        return self.FeedForward(
            _parent=self,
            input=autoencoder.input,
            output=classification.output,
            embedding=autoencoder.embedding,
            reconstruction=autoencoder.output,
            classification=classification.output
        )

    