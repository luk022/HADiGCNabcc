"""
loss_functions.py

Public loss interface for the HADiGCN package.
This file provides simple, user-facing wrappers for:

- TopKCrossEntropyLoss (re-exported from hadigcn.difftopk.losses)
- CrossEntropyLossWrapper (clean wrapper around torch.nn.CrossEntropyLoss)

Authoritative difftopk implementation remains in hadigcn/difftopk/losses.py.
This file provides a clean API so users import losses from:
    from hadigcn.loss_functions import TopKCrossEntropyLoss, CrossEntropyLossWrapper
"""

import torch
import torch.nn as nn


import sys  
sys.path.append(r"D:/Github repos/difftopk") ## Use your own directory that contains difftopk 


try:
    # Import TopKCrossEntropyLoss from the external 'difftopk' package (https://github.com/Felix-Petersen/difftopk/tree/main).
    # NOTE: Users must install this package separately.
    from difftopk import TopKCrossEntropyLoss
except ImportError:
    # Define a placeholder that raises a clear error if difftopk is missing.
    def TopKCrossEntropyLoss(*args, **kwargs):
        raise ImportError(
            "The TopKCrossEntropyLoss is not available because the 'difftopk' package "
            "is missing. Please install it separately from: https://github.com/Felix-Petersen/difftopk/tree/main"
        )


class CrossEntropyLossWrapper(nn.Module):
    """Simple wrapper around torch.nn.CrossEntropyLoss.

    This ensures that all losses used in the package follow the same API.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
