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
from abc import ABC, abstractmethod



network = type("EmptyNetwork", (Network,), {
    "logits": lambda _: tuple(),
    "device": lambda _: "cuda",
    "input_shape": lambda _: (28,28),
    "output_shape": lambda _: (10,)
})(logits=tuple(), device="auto", input_shape=(12,34), output_shape=(12,34))
network.save()
# %%
