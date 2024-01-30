# %%
from matplotlib.colors import Colormap


from xai.time import *
from xai.asteroids import *
from xai.window import *
from xai.action import *
from xai.angle import *
from torch.nn import Sequential, Linear, ReLU
from torch import tensor, Tensor
from xai.policy import *
from xai.optimizer import *
from torchvision.datasets.mnist import MNIST
from xai.loss import *
import matplotlib.pyplot as plt

import torch

# %%
def foo(arg):
    print(arg)

for i in range(5000):
    foo({
        "i": i,
        "txt": str(i)
    })

Losses.by_name("binary_cross_entropy").value(3,target=4,weight=3,size_average=5)

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

(Policy((28,28),10) + Policy(10,(28,28))).sgd().fit()
# %%
images = mnist.data

encoder = ContinuousPolicy.new((28,28),10, hidden_layers=2)
decoder = ContinuousPolicy.new(10,(28,28), hidden_layers=2)

encoder + decoder
# %%

Adam(encoder + decoder).sgd().fit(images, images, 5000, 32, verbose=True)

# %%
image = mnist.data[950]

# %%
reconstruction = (encoder + decoder)(images).tensor()
plt.imshow(image.cpu().numpy(), cmap="gray")
# %%
plt.imshow(reconstruction.detach().cpu().numpy(), cmap="gray")
# %%

classifier = DiscretePolicy.new(28*28,10, hidden_layers=3)
SGD(classifier).fit(images, labels, steps=5000, batch_size=32, verbose=True, lr=0.1)

# %%

classifier(image.flatten())



# %%



env = Asteroids()
env.play(show=True, translate=True, rotate=True, fps=60, stochastic=False)
# %%

isinstance((1,2,3), Iterable)
# %%

class Foo(NamedTuple):
    id: int
    name: str

    def hello(self):
        return self.id
    

id,name = Foo(4,"as")

