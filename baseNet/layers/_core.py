"""Implementations of multi-layer perceptron (MLP) and other helper classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor, nn
from torch.nn import LSTM, Linear, Module, ModuleList

if TYPE_CHECKING:
    from collections.abc import Sequence


class MLP(nn.Module):
    """An implementation of a multi-layer perceptron."""

    def __init__(
            self,
            dims: Sequence[int],
            activation: Callable[[Tensor], Tensor] | None = None,
            activate_last: bool = False,
            bias_last: bool = True,
    ) -> None:
        """
        Args:
            dims: Dimensions of each layer of MLP
            activation: Activation function
            activate_last: Whether to apply activation to last layer
            bias_last: Whether to apply bias to last layer
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))
                if activation is not None:
                    self.layers.append(activation)  # type: ignore
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))
                if activation is not None and activate_last:
                    self.layers.append(activation)  # type: ignore

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f'MLP({", ".join(dims)})'

    @property
    def last_linear(self) -> Linear | None:
        """
        Returns:
            The last linear layer
        """
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer
        raise RuntimeError

    @property
    def depth(self) -> int:
        """
        Returns:
            Depth of MLP
        """
        return self._depth

    @property
    def in_features(self) -> int:
        """
        Returns:
            Input features of MLP
        """
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        """
        Returns:
            Output features of MLP
        """
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs):
        """
        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x
