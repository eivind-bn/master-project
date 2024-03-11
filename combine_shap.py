from numpy.typing import NDArray
import numpy as np

digit_count = 10
latent_connections = 5

reconstruction_shap: NDArray[np.float64] = ... 
# shape: (latent_connections, image_height*image_width) = (5, 28*28) = (5, 784)

classifier_head_shap: NDArray[np.float64] = ... 
# shape: (latent_connections, digit_count) = (5, 10)

combined_shap = np.zeros((digit_count,784), dtype=np.float32)
# We want 10 shap values for every 784 pixel, i.e. of shape: (10, 784).

for i in range(digit_count):
    for j in range(latent_connections):
        # We find the reconstruction contribution for this particular latent connection.
        latent_contribution_on_pixels = reconstruction_shap[j]
        # (784,)

        # We assume that the contribution is reflexive.
        pixels_contribution_on_latent = latent_contribution_on_pixels
        # (784,)

        # We find the specific digit contribution from this specific latent connection.
        latent_contribution_on_digit = classifier_head_shap[j,i]
        # (1,)

        # We calculate how much the pixels impact the classification through one specific latent connection.
        pixels_contribution_on_digit_through_one_latent = pixels_contribution_on_latent * latent_contribution_on_digit
        # (784,) = (784,) * (1,)

        # We need to calculate and accumulate the result for each latent connection.
        combined_shap[i] = combined_shap[i] + pixels_contribution_on_digit_through_one_latent
        # (784,) = (784,) + (784,)
        

# Done!

# Can also be written more concisely as a transposed matrix multiplication:
combined_shap = classifier_head_shap.T @ reconstruction_shap
# (10, 784) = (5, 10)^T @ (5, 784) = (10, 5) @ (5, 784)