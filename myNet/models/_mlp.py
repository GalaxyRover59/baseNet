from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from myNet.layers import MLP
if TYPE_CHECKING:
    import dgl


class MLPNet(nn.Module):
    def __init__(
            self,
            dims,
            activation=None,
            activate_last: bool = False,
            bias_last: bool = True,
            n_layers: int = 3,
            **kwargs):
        super().__init__()
        self.MLPblock = nn.ModuleList(
            {
                MLP(dims, activation, activate_last, bias_last)
                for _ in range(n_layers)
            }
        )

    def forward(
            self,
            g,
            state_attr: torch.Tensor | None = None,
            l_g=None,
            return_all_layer_output: bool = False,
    ):
        """

        :param g:
        :param state_attr:
        :param l_g:
        :param return_all_layer_output:
        :return:
        """
        pass
