# %%

from xai.time import *
from xai.asteroids import *
from xai.window import *

# %%

a = Seconds.now() + Days(1)
b = Seconds.now()

a < b, b > a, b < a, a > b
# %%


with Window("Foo", fps=60, scale=2.0) as window:
    for i in range(400):
        frame = np.random.randint(low=0, high=255, size=(300,300,3), dtype=np.uint8)
        s = window(frame).match({
            "wa": lambda: print("a"),
            (1,2,3): lambda: None,
            None: lambda: 2
        })


# %%
env = Asteroids()
env.play(show=True, translate=True, rotate=True)
# %%
