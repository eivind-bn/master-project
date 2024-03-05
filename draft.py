# %%
import torch
from typing import *
from numpy import ndarray
from numpy.typing import NDArray
from torch import Tensor
from xai import Device
from xai.network import Network
import torch
import math

f1 = Network.dense((100,100), (3,), device="cuda")
f2 = Network.dense((3,), (100,100), device="cuda", output_activation="Sigmoid")

f1 + f2
# %%

(f1 + f2)(torch.ones((100,100)))

# %%
