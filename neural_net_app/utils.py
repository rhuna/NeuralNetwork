from __future__ import annotations
# Enables modern type hint behavior.

import random
# Python’s built-in random module.

import numpy as np
# NumPy random support.

import torch
# PyTorch random support and device utilities.


def set_seed(seed: int = 42) -> None:
    # Sets random seeds so training results are more reproducible.

    random.seed(seed)
    # Seeds Python's built-in random module.

    np.random.seed(seed)
    # Seeds NumPy's random generator.

    torch.manual_seed(seed)
    # Seeds PyTorch on CPU.

    if torch.cuda.is_available():
        # Checks whether CUDA/GPU is available.

        torch.cuda.manual_seed_all(seed)
        # Seeds all CUDA devices as well.


def get_device() -> torch.device:
    # Returns the best device available for PyTorch.

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If CUDA is available, use GPU.
    # Otherwise use CPU.