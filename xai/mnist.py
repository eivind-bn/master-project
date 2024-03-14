from . import *
from dataclasses import dataclass
from torch import Tensor

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

class MNIST(Generic[Sl]):
    @dataclass
    class FeedForward(Network.FeedForward):
        _parent:        "MNIST"
        embedding:      Lazy[Tensor]
        reconstruction: Lazy[Tensor]
        classification: Lazy[Tensor]
        
        def digits(self) -> Tuple[int,...]:
            return tuple(self.classification().reshape((-1,10)).argmax(dim=1))
    
        @overload
        def explain(self, 
                    domain: Literal["reconstruction"], 
                    explainer: Explainers, 
                    verbose: bool = ...) -> Explanation[Sl,Sx]: ...

        @overload
        def explain(self, 
                    domain: Literal["embedding classification"], 
                    explainer: Explainers, 
                    verbose: bool = ...) -> Explanation[Sl,Sy]: ...
        @overload
        def explain(self, 
                    domain: Literal["image classification", "reconstruction classification"], 
                    explainer: Explainers, 
                    verbose: bool = ...) -> Explanation[Sx,Sy]: ...

        def explain(self, 
                    domain: Domain, 
                    explainer: Explainers, 
                    verbose: bool = False) -> Explanation:
            match domain:
                case "reconstruction":
                    return self._parent.reconstruction_explainer(explainer).explain(self.embedding(), verbose)[0]
                case "embedding classification":
                    return self._parent.classifier_head_explainer(explainer).explain(self.embedding(), verbose)[0]
                case "image classification":
                    return self._parent.explainer(explainer).expl
                case _:
                    assert_never(domain)

    def __init__(self, 
                 latent_shape:      Sl,
                 hidden_layers:     int|Sequence[int] = 2,
                 hidden_activation: Activation|None = "ReLU",
                 device:            Device = "auto") -> None:
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
            hidden_activation=hidden_activation,
            output_activation="Sigmoid",
            device=device
        )

        self.classifier_head = Network.dense(
            input_dim=self.autoencoder.latent_shape,
            output_dim=self.output_shape,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation,
            output_activation=lambda type: type.Softmax(dim=1),
            device=device
        )

        self._explainers: Dict[Domain,Dict[Explainers,Explainer[Sx|Sl,Sx|Sy]]] = {}

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
    
    def get_sample(self, digit: int) -> Tuple[Tensor,Tensor]:
        idx = random.sample(self._val_indices_by_digit[digit], k=1)[0]
        return self.val_data[idx], self.val_labels[idx]
    
    @overload
    def explainer(self, type: Explainers, domain: Literal["reconstruction"]) -> Explainer[Sl,Sx]: ...

    @overload
    def explainer(self, type: Explainers, domain: Literal["embedding classification"]) -> Explainer[Sl,Sy]: ...

    @overload
    def explainer(self, type: Explainers, domain: Literal["image classification", "reconstruction classification"]) -> Explainer[Sx,Sy]: ...

    def explainer(self, type: Explainers, domain: Domain) -> Explainer:
        explainer = self._explainers.setdefault(domain, {}).get(type)
        if explainer is None:
            match domain:
                case "reconstruction":
                    explainer = self.autoencoder.decoder.explainer(type, self(self.val_data).embedding())
                case "embedding classification":
                    explainer = self.classifier_head.explainer(type, self(self.val_data).embedding())
                case "image classification":
                    explainer = (self.autoencoder.encoder + self.classifier_head).explainer(type, self.val_data)
                case "reconstruction classification":
                    explainer = (self.autoencoder.encoder + self.classifier_head).explainer(type, self.val_data)
                case _:
                    assert_never(domain)
        else:
            return explainer
    
    def reconstruction_explainer(self, type: Explainers) -> Explainer[Sl,Sx]:
        explainer = self._explainers.get(type)
        if explainer is None:
            explainer = self.autoencoder.decoder.explainer(type, self(self.val_data).embedding())
            self._explainers[type] = explainer

        return explainer
    
    def classifier_head_explainer(self, type: Explainers) -> Explainer[Sl,Sy]:
        explainer = self._classifier_head_explainers.get(type)
        if explainer is None:
            explainer = self.classifier_head.explainer(type, self(self.val_data).embedding())
            self._classifier_head_explainers[type] = explainer

        return explainer
    
    def classifier_head_explainer(self, type: Explainers) -> Explainer[Sl,Sy]:
        explainer = self._classifier_head_explainers.get(type)
        if explainer is None:
            explainer = self.classifier_head.explainer(type, self(self.val_data).embedding())
            self._classifier_head_explainers[type] = explainer

        return explainer
    
    def fit(self, 
            epochs: int,
            batch_size: int,
            loss_criterion: Loss = "MSELoss",
            early_stop_cont: int|None = 10,
            verbose: bool = False,
            info: str | None = None) -> TrainStats:
          self._explainers.clear()
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
                        info: str | None = None) -> TrainStats:
          self._explainers.clear()
          self._classifier_head_explainers.clear()
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
                    info: str | None = None) -> TrainStats:
          self._explainers.clear()
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
                            info: str | None = None) -> TrainStats:
         self._classifier_head_explainers.clear()
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
                       info: str | None = None) -> TrainStats:
        self._explainers.clear()
        self._classifier_head_explainers.clear()
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

    def __call__(self, X: Array|Lazy[Array]) -> "MNIST.FeedForward":
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

    