from . import *
from dataclasses import dataclass
from plotly.graph_objects import Figure # type: ignore

import plotly.express as px # type: ignore

class Epoch(NamedTuple):
    batch_size: int
    train_loss: float
    val_loss:   float|None
    accuracy:   float|None
    info:       str|None

class Milestone:

    def __init__(self, info: str|None = None) -> None:
        self.epochs: List[Epoch] = []
        info = info

class TrainHistory:

    def __init__(self) -> None:
        self.milestones: List[Milestone] = []

    def __enter__(self, info: str|None = None) -> Callable[[Epoch],None]:
        milestone = Milestone(info=info)
        self.milestones.append(milestone)
        return milestone.epochs.append
    
    def __exit__(self, *args, **kwargs) -> None:
        pass

    def append_epoch(self,
                     batch_size: int,
                     train_loss: float,
                     val_loss:   float|None = None,
                     accuracy:   float|None = None,
                     info:       str|None = None) -> None:
        self.append(Epoch(
            batch_size=batch_size,
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=accuracy,
            info=info
        ))

    def figure(self, 
               info: str|None = None, 
               batch_size: bool = False) -> Figure:
        
        for epoch,record in enumerate(self):
            pass

        epochs_name = "epochs"
        train_loss_name = "train-loss"
        y_trends: List[str] = [train_loss_name]
        stats: Dict[str,Iterable[int|float]] = {
            epochs_name:       range(len(self.train_losses)),
            train_loss_name:   self.train_losses,
            }
        
        if self.val_losses is not None:
            validation_loss = "validation-loss"
            y_trends.append(validation_loss)
            stats[validation_loss] = self.val_losses

        if self.accuracies:
            accuracies_name = "accuracies"
            y_trends.append(accuracies_name)
            stats[accuracies_name] = self.accuracies

        if batch_size:
            batch_size_name = "batch-size"
            y_trends.append(batch_size_name)
            stats[batch_size_name] = self.batch_size

        return px.line(stats, x=epochs_name, y=y_trends, title=info if info else self.info).add_li