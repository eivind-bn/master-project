from typing import *
from .policy import Policy, Device
from .buffer import Buffer
from .bytes import GigaBytes
from .explainer import Explainer
from .stats import TrainStats
from .feed_forward import FeedForward
from torch import Tensor
from numpy.typing import NDArray
from numpy import float32, uint8

import mnist
import numpy as np
import torch

PredictType = Literal[
    "Latent",
    "Reconstruction", 
    "Digit"]

class MNIST:

    def __init__(self, 
                 latent_dim: int, 
                 scale: int, 
                 device: Device, 
                 use_ram: bool|None = None, 
                 verbose: bool = True) -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_ram is None:
            use_ram = scale < 3

        self.image_dim = tuple(scale*dim for dim in (28,28))
        self.latent_dim = latent_dim

        def create_mnist() -> Iterator[Tensor]:
            images = mnist.train_images().astype(np.float32) / 255.0
            if scale > 1:
                image: np.ndarray
                for image in images:
                    image = image.repeat(scale, axis=0).repeat(scale, axis=1)
                    yield torch.from_numpy(image).cpu()
            else:
                yield torch.from_numpy(images).cpu()

        self.buffer = Buffer(
            entries=create_mnist(),
            eviction_policy="Random",
            use_ram=use_ram,
            max_memory=GigaBytes(5),
            verbose=verbose
        )

        self.labels = torch.from_numpy(mnist.train_labels())
        labels_one_hot = torch.zeros((self.labels.shape[0],10)).float()
        labels_one_hot[torch.arange(0,self.labels.shape[0]),self.labels.int()] = 1.0

        if scale > 3:
            hidden_layers: List[int]|int = [2**8, 2**6]
        else:
            hidden_layers = 2

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

    @overload
    def predict(self, 
                image:      NDArray[uint8|float32]|Tensor, 
                type:       Literal["Reconstruction", "Latent"]) -> Tuple[FeedForward, Explainer]: ...
    
    @overload
    def predict(self, 
                image:      NDArray[uint8|float32]|Tensor,
                type:       Literal["Digit"]) -> Tuple[int, Explainer]: ...
    
    @overload
    def predict(self, 
                image:      NDArray[uint8|float32]|Tensor, 
                type:       PredictType) -> Tuple[FeedForward|int, Explainer]:
        image = self.to_tensor(image)

        match type:
            case "Digit":
                network = self.autoencoder
                Y_hat = network.predict(image)
                return int(Y_hat.tensor(True).argmax().item()), Explainer(
                    network=network,
                    feed_forward=Y_hat
                )
            case "Latent":
                network = self.encoder
                Y_hat = network.predict(image)
            case "Reconstruction":
                network = self.autoencoder
                Y_hat = network.predict(image)

        explainer = Explainer(network=network, feed_forward=Y_hat)
                
        return Y_hat, explainer
    