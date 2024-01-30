from typing import *
from abc import ABC, abstractmethod
from enum import *
from dataclasses import dataclass
from typing import Any
from torch import Tensor

import torch.nn.functional as F

class LossFunction(Protocol):
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor: ...

LossType = Literal[
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "poisson_nll_loss"
    "cosine_embedding_loss",
    "cross_entropy",
    "ctc_loss",
    "gaussian_nll_loss",
    "hinge_embedding_loss",
    "kl_div",
    "l1_loss",
    "mse_loss",
    "margin_ranking_loss",
    "multilabel_margin_loss",
    "multilabel_soft_margin_loss",
    "multi_margin_loss",
    "nll_loss",
    "huber_loss",
    "smooth_l1_loss",
    "soft_margin_loss",
    "triplet_margin_loss",
    "triplet_margin_with_distance_loss"
]


class Losses(Enum):
    binary_cross_entropy = member(F.binary_cross_entropy)
    binary_cross_entropy_with_logits = Loss(F.binary_cross_entropy_with_logits)
    poisson_nll_loss = Loss(F.poisson_nll_loss)
    cosine_embedding_loss = Loss(F.cosine_embedding_loss)
    cross_entropy = Loss(F.cross_entropy)
    ctc_loss = Loss(F.ctc_loss)
    gaussian_nll_loss = Loss(F.gaussian_nll_loss)
    hinge_embedding_loss = Loss(F.hinge_embedding_loss)
    kl_div = Loss(F.kl_div)
    l1_loss = Loss(F.l1_loss)
    mse_loss = Loss(F.mse_loss)
    margin_ranking_loss = Loss(F.margin_ranking_loss)
    multilabel_margin_loss = Loss(F.multilabel_margin_loss)
    multilabel_soft_margin_loss = Loss(F.multilabel_soft_margin_loss)
    multi_margin_loss = Loss(F.multi_margin_loss)
    nll_loss = Loss(F.nll_loss)
    huber_loss = Loss(F.huber_loss)
    smooth_l1_loss = Loss(F.smooth_l1_loss)
    soft_margin_loss = Loss(F.soft_margin_loss)
    triplet_margin_loss = Loss(F.triplet_margin_loss)
    triplet_margin_with_distance_loss = Loss(F.triplet_margin_with_distance_loss)

    @classmethod
    def by_name(cls, name: LossType) -> "Losses":
        return getattr(cls, name)
    
Losses._member_names_