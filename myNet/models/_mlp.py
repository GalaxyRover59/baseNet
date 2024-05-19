from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from myNet.graph.compute import compute_pair_vector_and_distance
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
        self.dims = dims

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
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist
        bond_embed = MLP([len(bond_vec), self.dims[0]])
        for block in self.MLPblock:
            out = block(bond_embed)
            bond_vec = MLP([len(out), self.dims[0]])
            if self.dropout:
                bond_vec = self.dropout(bond_vec)
        out = torch.squeeze(MLP([self.dims[0], 1]))
