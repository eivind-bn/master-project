from . import *

import torch

Device: TypeAlias = Literal["cpu", "cuda", "cuda:0", "cuda:1", "auto"]|torch.device

def get_device(device: Device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    elif device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)