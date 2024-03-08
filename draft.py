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
from xai.autoencoder import AutoEncoder
from xai.mnist import MNIST
import matplotlib.pyplot as plt

mnist = MNIST((5,), hidden_activation="LeakyReLU")
mnist

# %%
mnist.fit_autoencoder(3000, 128, "MSELoss", verbose=True).plot_loss()
# %%
mnist.fit_classifier(3000, 128, "CrossEntropyLoss", verbose=True).plot_loss()

# %%

X = mnist.train_data[750]
Y = mnist(X)
image = torch.hstack([X,Y.reconstruction()]).cpu()

Y.digits(), plt.imshow(image.numpy(force=True), cmap="gray")
# %%

