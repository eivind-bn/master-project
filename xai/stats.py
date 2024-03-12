from typing import *
from dataclasses import dataclass
from plotly.graph_objects import Figure # type: ignore

import plotly.express as px # type: ignore

@dataclass(frozen=True)
class TrainStats:
    batch_size:     Tuple[int,...]
    train_losses:   Tuple[float,...]
    val_losses:     Tuple[float,...]|None
    accuracies:     Tuple[float,...]|None
    info:           str|None

    def plot_loss(self, 
                  info: str|None = None, 
                  batch_size: bool = False) -> Figure:
        epochs_name = "epochs"
        train_loss_name = "train-loss"
        y_trends: List[str] = [train_loss_name]
        stats: Dict[str,Iterable[int|float]] = {
            epochs_name:       range(len(self.train_losses)),
            train_loss_name:   self.train_losses
            }
        
        if self.val_losses is not None:
            validation_loss = "validation-loss"
            y_trends.append(validation_loss)
            stats[validation_loss] = self.val_losses

        if self.accuracies:
            accuracies_name = "accuracies_name"
            y_trends.append(accuracies_name)
            stats[accuracies_name] = self.accuracies

        if batch_size:
            batch_size_name = "batch-size"
            y_trends.append(batch_size_name)
            stats[batch_size_name] = self.batch_size

        return px.line(stats, x=epochs_name, y=y_trends, title=info if info else self.info)      
    
    def __getitem__(self, slice: slice) -> "TrainStats":
        return TrainStats(
            batch_size=self.batch_size,
            train_losses=self.train_losses[slice],
            val_losses=None if self.val_losses is None else self.val_losses[slice],
            accuracies=None if self.accuracies is None else self.accuracies[slice],
            info=self.info
        )