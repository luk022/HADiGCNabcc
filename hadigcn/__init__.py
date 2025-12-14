"""
HADiGCN_thesis

Public API exports:
- Models (HADiGCN, MLP1Layer, MLP2Layer)
- Loss functions (TopKCrossEntropyLoss, CrossEntropyLossWrapper)
- Training utilities (train_one_epoch, validate_one_epoch, train)
- General utilities (set_seed, get_device)
"""

# -----------------------------
# Models
# -----------------------------
from .models import HADiGCN, MLP1Layer, MLP2Layer

# -----------------------------
# Losses
# -----------------------------
from .loss_functions import (
    TopKCrossEntropyLoss,        
    CrossEntropyLossWrapper,
)

# -----------------------------
# Training utilities
# -----------------------------
from .train import (
    train_one_epoch,
    validate_one_epoch,
    train,
)

# -----------------------------
# General utilities
# -----------------------------
from .utils import (
    set_seed,
    get_device,
    findDuplicated,
    high_var_npdata,
    sparse_mx_to_torch_sparse_tensor,
    process_features_and_adj,
)

__all__ = [
    # models
    "HADiGCN",
    "MLP1Layer",
    "MLP2Layer",

    # losses
    "TopKCrossEntropyLoss",
    "CrossEntropyLossWrapper",

    # training utils
    "train_one_epoch",
    "validate_one_epoch",
    "train",

    # General utilities
    "set_seed",
    "get_device",
    "findDuplicated",
    "high_var_npdata",
    "sparse_mx_to_torch_sparse_tensor",
    "process_features_and_adj",
]