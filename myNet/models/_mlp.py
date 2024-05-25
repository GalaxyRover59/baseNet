from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from myNet.graph.compute import compute_pair_vector_and_distance
from myNet.layers import MLP

# if TYPE_CHECKING:
import dgl


class MLPNet(nn.Module):
    def __init__(
            self,
            dims,
            activation=None,
            activate_last: bool = False,
            bias_last: bool = True,
            n_layers: int = 3,
            dropout: float = 0.0,
            **kwargs):
        super().__init__()
        self.MLPblock = nn.ModuleList(
            {
                MLP(dims, activation, activate_last, bias_last)
                for _ in range(n_layers)
            }
        )
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.init0 = MLP([3, dims[0]])
        self.init1 = MLP([dims[-1], dims[0]])
        self.out = MLP([dims[0], 1])

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
        bond_embed = self.init0(dgl.readout_edges(g, "bond_vec", op='mean'))
        for block in self.MLPblock:
            out = block(bond_embed)
            bond_embed = self.init1(out)
            if self.dropout:
                bond_embed = self.dropout(bond_embed)
        out = torch.squeeze(self.out(bond_embed))
        
        return out
