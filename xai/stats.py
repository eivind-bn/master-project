from typing import *
from dataclasses import dataclass
from plotly.graph_objects import Figure # type: ignore

import plotly.express as px # type: ignore

@dataclass(frozen=True)
class TrainStats:
    batch_size: int
    losses:     Tuple[float,...]
    info:       str|None

    def plot_loss(self, info: str|None = None) -> Figure:
        return px.line({
            "epochs": range(len(self.losses)),
            "loss"  : self.losses
        }, x="epochs", y=["loss"], title=self.info if info is None else info)
    
    def __getitem__(self, slice: slice) -> "TrainStats":
        return TrainStats(
            batch_size=self.batch_size,
            losses=self.losses[slice],
            info=self.info
        )