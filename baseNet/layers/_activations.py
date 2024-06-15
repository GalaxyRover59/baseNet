"""Custom activation functions."""

from __future__ import annotations

from enum import Enum

from torch import nn


class ActivationFunction(Enum):
    """Enumeration of optional activation functions."""

    swish = nn.SiLU
    sigmoid = nn.Sigmoid
    tanh = nn.Tanh
    softplus = nn.Softplus
