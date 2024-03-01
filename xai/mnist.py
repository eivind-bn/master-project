from typing import *
from .policy import Policy
from .reflist import RefList
from .bytes import GigaBytes
from .explainer import Explainer
from .stats import TrainStats
from .feed_forward import FeedForward
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, uint8
from . import Device, get_device

import mnist
import numpy as np
import torch

PredictType = Literal[
    "Latent",
    "Reconstruction", 
    "Digit"]

class MNIST:

    def __init__(self, 
                 latent_dim:    int, 
                 device:        Device) -> None:
        
        device = get_device(device)

        self.latent_dim = latent_dim

        self.labels = torch.from_numpy(mnist.train_labels())
        labels_one_hot = torch.zeros((self.labels.shape[0],10)).float()
        labels_one_hot[torch.arange(0,self.labels.shape[0]),self.labels.int()] = 1.0

        self.encoder = Policy.new(
            input_dim=self.image_dim,
            output_dim=self.latent_dim,
            hidden_layers=hidden_layers,
            device=self.device
        )

        self.decoder = Policy.new(
            input_dim=self.encoder.output_dim,
            output_dim=self.encoder.input_dim,
            hidden_layers=hidden_layers,
            device=self.device
        )

        self.classifier_head = Policy.new(
            input_dim=self.encoder.output_dim,
            output_dim=10,
            device=self.device
        )

        self.autoencoder = self.encoder + self.decoder
        self.classifier = self.encoder + self.classifier_head

    def to_tensor(self, array: NDArray[uint8|float32]|Tensor) -> Tensor:
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device=self.device, dtype=torch.float32)
        else:
            return array.to(device=self.device, dtype=torch.float32)