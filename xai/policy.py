from typing import *
from typing import Any
from torch import Tensor
from torch.nn import Sequential
from torch.optim import Optimizer, SGD, Adam
from .feed_forward import FeedForward

import torch

Device = Literal["cpu", "cuda", "cuda:0", "cuda:1", "auto"]
OptimizerType = Literal["sgd", "adam"]

class SgdParams(TypedDict, total=False):
    lr: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool

class AdamParams(TypedDict, total=False):
    lr: float|Tensor
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    foreach: bool
    maximize: bool
    capturable: bool
    differentiable: bool
    fused: bool

OptimizerParams = SgdParams|AdamParams

class Policy:

    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: int|Sequence[int] = 2,
                 optimizer_params: OptimizerParams|None = None,
                 device: Device = "auto") -> None:
        super().__init__()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._net = self._build_network(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            device=self.device
        )

        self._optimizers: Dict[OptimizerType,Optimizer] = {
            "sgd": torch.optim.SGD(self._net.parameters(), lr=0.1),
            "adam": torch.optim.Adam(self._net.parameters())
        }

        self._set_optim_param()

        self._loss_functions: Dict[LossType,object] = {
            "mse": torch.nn.MSELoss()
        }
        
    def predict(self, 
                x:              Tensor|FeedForward, 
                move_to_device: bool = False) -> FeedForward:
        
        if isinstance(x, FeedForward):
            x = x._output

        if move_to_device:
            x = x.to(device=self.device)

        return FeedForward(
            input=x,
            network=self._net
        )
    
    def __call__(self, x: Tensor|FeedForward) -> FeedForward:
        return self.predict(x)
    
    def fit(self, 
            X:                  Tensor, 
            Y:                  Tensor, 
            steps:              int, 
            batch_size:         int,
            optimizer_type:     OptimizerType, 
            optimizer_params:   OptimizerParams|None = None) -> None:
        
        optimizer = self._optimizers[optimizer_type]

        if optimizer_params:
            self._set_optim_param(optimizer, optimizer_params)
        
        for step in range(steps):
            optimizer.zero_grad()
            x,y = self._mini_batch(X, Y, batch_size)
            y_hat = self(x).tensor()
            loss = torch.nn.functional.mse_loss(y_hat, y)
            print(loss)
            loss.backward()
            optimizer.step()
            
    def save(self, path: str) -> None:
        torch.save(self, path)

    @staticmethod
    def _build_network(input_dim: int,
                       output_dim: int,
                       hidden_layers: int|Sequence[int] = 2,
                       device: Device = "auto") -> Sequential:

        net = torch.nn.Sequential()

        I, O = input_dim, output_dim

        if isinstance(hidden_layers, int):
            N = hidden_layers + 1
            layers = [I] + [int(I - ((I - O)*n)/(N)) for n in range(1, hidden_layers+1)] + [output_dim]
        elif isinstance(hidden_layers, Sequence):
            layers = [I] + list(hidden_layers) + [O]
        else:
            raise TypeError(f"Argument for layers has incompatible type: {type(hidden_layers)}")
        
        for i in range(1, len(layers)-1):
            net += torch.nn.Sequential(
                torch.nn.Linear(layers[i-1], layers[i], device=device),
                torch.nn.ReLU()
            )

        net += torch.nn.Sequential(torch.nn.Linear(layers[-2], layers[-1], device=device))

        return net
    
    def _mini_batch(self,
                    X: Tensor, 
                    Y: Tensor, 
                    batch_size: int, 
                    sample_dim: int = 0) -> Tuple[Tensor,Tensor]:
        assert X.shape[0] == Y.shape[0] and batch_size > 0
        idx = torch.randperm(min(batch_size, X.shape[sample_dim]), device=self.device)
        return X.index_select(sample_dim, idx), Y.index_select(sample_dim, idx)
    
    @staticmethod
    def _set_optim_param(optimizer: Optimizer, params: OptimizerParams) -> None:
        for param_group in optimizer.param_groups:
            for param,value in params.items():
                param_group[param] = value

    @staticmethod
    def load(path: str) -> "Policy"|NoReturn:
        policy: Policy = torch.load(path)
        if isinstance(policy, Policy):
            return policy
        else:
            raise TypeError(f"Unpickled object is not a Policy, but of type: {type(policy)}")
        
    def __repr__(self) -> str:
        return str(self._net)
