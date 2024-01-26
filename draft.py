# %%
from matplotlib.colors import Colormap


from xai.time import *
from xai.asteroids import *
from xai.window import *
from xai.action import *
from xai.angle import *
from torch.optim import Optimizer, SGD, Adam
from torch.nn import Sequential, Linear, ReLU
from torch import tensor, Tensor
from xai.tensor import *
from xai.policy import Policy
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
# %%
mnist = MNIST(".", download=True)
mnist.data = mnist.data.float()
mnist.data /= 255.0
mnist.data = mnist.data.cuda()
mnist.data
# %%

labels = torch.zeros((mnist.data.shape[0],10)).float()
labels[torch.arange(0,mnist.data.shape[0]),mnist.targets] = 1.0
labels = labels.cuda()
labels


# %%

f = Policy(28*28,10, device="auto")
f.fit(mnist.data.flatten(start_dim=1), labels, 5000, 32, "adam", optimizer_params={
})
# %%

plt.imshow(mnist.data[600].cpu().numpy()), f(mnist.data[600].flatten())
# %%


env = Asteroids()
env.play(show=True, translate=False, rotate=False, fps=60, stochastic=False)
# %%
# %%
