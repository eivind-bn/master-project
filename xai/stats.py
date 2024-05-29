from . import *
from dataclasses import dataclass

import matplotlib.pyplot as plt

class Epoch(TypedDict):
    batch_size: int
    train_loss: float
    val_loss:   float|None
    accuracy:   float|None

class Milestone:

    def __init__(self) -> None:
        self.epochs: List[Epoch] = []

class TrainHistory:

    def __init__(self) -> None:
        self.milestones: List[Milestone] = []

    def __enter__(self) -> Callable[[Epoch],None]:
        milestone = Milestone()
        self.milestones.append(milestone)
        return milestone.epochs.append
    
    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        return

    def figure(self, 
               info: str|None = None, 
               batch_size: bool = False) -> None:
        
        for i,milestone in enumerate(self.milestones):
            plt.figure(dpi=250)
            x = list(range(len(milestone.epochs)))

            zipped_values: Dict[str,List[Any]] = {}
            for epoch in milestone.epochs:
                for field,value in epoch.items():
                    zipped_values.setdefault(field, []).append(value)

            for field,values in zipped_values.items():
                if not batch_size and field == "batch_size":
                    continue

                plt.plot(x, values, label=field)
                plt.legend()

            if info:
                plt.title(info)
                
                    
            