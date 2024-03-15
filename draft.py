# # %%
# import torch
# from typing import *
# from numpy import ndarray
# from numpy.typing import NDArray
# from torch import Tensor
# from xai import Device
# from xai.network import Network
# import torch
# import math
# from abc import ABC, abstractmethod
# from xai.autoencoder import AutoEncoder
# from xai.mnist import MNIST
# import matplotlib.pyplot as plt

# mnist = MNIST((5,), hidden_activation="LeakyReLU")
# mnist

# # %%
# mnist.fit_autoencoder(3000, 128, "MSELoss", verbose=True).plot_loss()
# # %%
# mnist.fit_classifier_head(3000, 128, "CrossEntropyLoss", verbose=True).plot_loss()

# # %%

# X = mnist.train_data[750]
# Y = mnist(X)
# image = torch.hstack([X,Y.reconstruction()]).cpu()

# Y.digits(), plt.imshow(image.numpy(force=True), cmap="gray")
# # %%

# from numpy.typing import NDArray
# import numpy as np

# digit_count = 10
# latent_connections = 5

# reconstruction_shap: NDArray[np.float64] = ... 
# # shape: (latent_connections, image_height*image_width) = (5, 28*28) = (5, 784)

# classifier_head_shap: NDArray[np.float64] = ... 
# # shape: (latent_connections, digit_count) = (5, 10)

# combined_shap = np.zeros((digit_count,784), dtype=np.float32)
# # We want 10 shap values for every 784 pixel, i.e. of shape: (10, 784).

# for i in range(digit_count):
#     for j in range(latent_connections):
#         latent_contribution_on_pixels = reconstruction_shap[j]
#         # (784,)

#         # We assume that the importance is reflexive.
#         pixels_contribution_on_latent = latent_contribution_on_pixels
#         # (784,)

#         latent_contribution_on_digit = classifier_head_shap[j,i]
#         # (1,)

#         # We calculate how much the pixels impact the classification through one specific latent connection.
#         pixels_contribution_on_digit_through_one_latent = pixels_contribution_on_latent * latent_contribution_on_digit

#         # We need to calculate and accumulate the result for each latent connection.
#         combined_shap[i] = combined_shap[i] + pixels_contribution_on_digit_through_one_latent
#         # (784,) = (784,) * (1,)


# # Can also be written more concisely as a transposed matrix multiplication:
# combined_shap = classifier_head_shap.T @ reconstruction_shap
# # (10, 784) = (5, 10)^T @ (5, 784) = (10, 5) @ (5, 784)