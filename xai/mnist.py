from . import *
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Module

import mnist # type: ignore
import torch
import random

Sx: TypeAlias = Tuple[Literal[28],Literal[28]]
Sy: TypeAlias = Tuple[Literal[10]]
Sl = TypeVar("Sl", bound=Tuple[int,...])
Domain: TypeAlias = Literal[
    "reconstruction", 
    "embedding classification", 
    "image classification", 
    "reconstruction classification"
    ]

@dataclass
class MNISTFeedForward(Generic[Sl], FeedForward[Sx,Sy]):
    parent:         "MNIST[Sl]"
    embedding:      FeedForward[Sx,Sl]
    reconstruction: FeedForward[Sl,Sx]
    classification: FeedForward[Sl,Sy]
    
    def digits(self) -> Tuple[int,...]:
        return tuple(self.classification().reshape((-1,10)).argmax(dim=1))

class MNIST(Generic[Sl], Network[Sx,Sy]):


    def __init__(self, 
                 latent_shape:                      Sl,
                 hidden_layers:                     int|Sequence[int] = 2,
                 autoencoder_hidden_activation:     Activation|None = "ReLU",
                 autoencoder_output_activation:     Activation|None = "Sigmoid",
                 classifier_head_hidden_Activation: Activation|None = "ReLU",
                 classifier_head_output_activation: Activation|None = "Softmax",
                 device:                            Device = "auto") -> None:
        super().__init__()

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

        self._val_indices_by_digit: Dict[int,List[int]] = {}

        for i,idx in enumerate(self.val_labels):
            self._val_indices_by_digit.setdefault(idx.item(), []).append(i)

        self.autoencoder = AutoEncoder(
            data_shape=self.input_shape,
            latent_shape=latent_shape,
            hidden_layers=hidden_layers,
            hidden_activation=autoencoder_hidden_activation,
            output_activation=autoencoder_output_activation,
            device=device
        )

        if classifier_head_output_activation == "Softmax":
            classifier_head_output_activation = lambda type: type.Softmax(dim=1)

        self.classifier_head = Network.dense(
            input_dim=self.autoencoder.latent_shape,
            output_dim=self.output_shape,
            hidden_layers=hidden_layers,
            hidden_activation=classifier_head_hidden_Activation,
            output_activation=classifier_head_output_activation,
            device=device
        )

    @property
    def modules(self) -> Tuple[Module,...]:
        return (self.autoencoder.encoder + self.classifier_head).modules
            
    @property
    def device(self) -> Device:
        return self._device
   
    @property
    def input_shape(self) -> Sx:
        return (28,28)
    
    @property
    def output_shape(self) -> Sy:
        return (10,)
    
    def get_sample(self, digit: int) -> Tuple[Tensor,Tensor]:
        idx = random.sample(self._val_indices_by_digit[digit], k=1)[0]
        return self.val_data[idx], self.val_labels[idx]
    
    def fit(self, 
            epochs: int,
            batch_size: int,
            loss_criterion: Loss = "MSELoss",
            early_stop_cont: int|None = 10,
            verbose: bool = False,
            info: str | None = None) -> TrainHistory:
          return self.autoencoder.decoder.adam().fit(
              X_train=self(self.train_data).embedding(),
              Y_train=self.train_data,
              X_val=self(self.val_data).embedding(),
              Y_val=self.val_data,
              early_stop_count=early_stop_cont,
              epochs=epochs,
              batch_size=batch_size,
              loss_criterion=loss_criterion,
              verbose=verbose,
              info=info
              )
    
    def fit_autoencoder(self, 
                        epochs: int,
                        batch_size: int,
                        loss_criterion: Loss = "MSELoss",
                        early_stop_cont: int|None = 10,
                        verbose: bool = False,
                        info: str | None = None) -> TrainHistory:
          return self.autoencoder.adam().fit(
             X_train=self.train_data,
             Y_train=self.train_data,
             X_val=self.val_data,
             Y_val=self.val_data,
             early_stop_count=early_stop_cont,
             epochs=epochs,
             batch_size=batch_size,
             loss_criterion=loss_criterion,
             verbose=verbose,
             info=info
         )
    
    def fit_decoder(self, 
                    epochs: int,
                    batch_size: int,
                    loss_criterion: Loss = "MSELoss",
                    early_stop_cont: int|None = 10,
                    verbose: bool = False,
                    info: str | None = None) -> TrainHistory:
          return self.autoencoder.decoder.adam().fit(
             X_train=self(self.train_data).embedding(),
             Y_train=self.train_data,
             X_val=self(self.val_data).embedding(),
             Y_val=self.val_data,
             early_stop_count=early_stop_cont,
             epochs=epochs,
             batch_size=batch_size,
             loss_criterion=loss_criterion,
             verbose=verbose,
             info=info
         )
    
    def fit_classifier_head(self, 
                            epochs: int,
                            batch_size: int,
                            early_stop_cont: int|None = 10,
                            verbose: bool = False,
                            info: str | None = None) -> TrainHistory:
         return self.classifier_head.adam().fit(
             X_train=self(self.train_data).embedding(),
             Y_train=self.train_labels,
             X_val=self(self.val_data).embedding(),
             Y_val=self.val_labels,
             early_stop_count=early_stop_cont,
             is_correct=lambda y_hat,y: y_hat.argmax(dim=1) == y,
             epochs=epochs,
             batch_size=batch_size,
             loss_criterion=lambda type: type.NLLLoss(),
             verbose=verbose,
             info=info
         )
    
    def fit_classifier(self,                            
                       epochs: int,
                       batch_size: int,
                       early_stop_cont: int|None = 10,
                       verbose: bool = False,
                       info: str | None = None) -> TrainHistory:
        return (self.autoencoder.encoder + self.classifier_head).adam().fit(
            X_train=self.train_data,
            Y_train=self.train_labels,
            X_val=self.val_data,
            Y_val=self.val_labels,
            early_stop_count=early_stop_cont,
            epochs=epochs,
            batch_size=batch_size,
            loss_criterion=lambda type: type.NLLLoss(),
            verbose=verbose,
            info=info
        )

    def __call__(self, X: Array|Lazy[Array]|"MNIST.FeedForward") -> "FeedForward":
        autoencoding = self.autoencoder(X)
        classification = self.classifier_head(autoencoding.embedding)
        return MNISTFeedForward(
            parent=self,
            input=autoencoding.input,
            output=classification.output,
            embedding=autoencoding.embedding,
            reconstruction=autoencoding.reconstruction,
            classification=classification
        )

    