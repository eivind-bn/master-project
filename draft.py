# %%
from matplotlib.colors import Colormap


from xai.time import *
from xai.asteroids import *
from xai.window import *
from xai.action import *
from xai.angle import *

# %%
env = Asteroids()
env.play(show=True, translate=False, rotate=False, fps=60, stochastic=False)
# %%
Actions.NOOP
# %%

l = list(AngleStates)
l
# %%


x = Radians(2) + Radians(4)
x += Radians(2)
x
# %%
