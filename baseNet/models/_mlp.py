from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from baseNet import DEFAULT_ELEMENTS
from baseNet.layers import MLP

# if TYPE_CHECKING:
import dgl
from baseNet.graph.converters import GraphConverter


class MLPNet(nn.Module):
    def __init__(
            self,
            dims,
            activation=None,
            activate_last: bool = False,
            bias_last: bool = True,
            n_layers: int = 3,
            dropout: float = 0.0,
            element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
            cutoff: float = 4.0,
            **kwargs):
        super().__init__()
        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff
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
            state_attr,
    ):
        """
        Args:
            g: dgl Graph
            state_attr: State attribute

        Returns:
            output: Output property for a batch of graphs
        """
        bond_embed = self.init0(dgl.readout_edges(g, "bond_vec", op='mean'))
        for block in self.MLPblock:
            out = block(bond_embed)
            bond_embed = self.init1(out)
            if self.dropout:
                bond_embed = self.dropout(bond_embed)
        out = torch.squeeze(self.out(bond_embed))

        return out

    def predict_structure(
            self,
            structure,
            state_attr: torch.Tensor | None = None,
            graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_attr: Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output: Output property
        """
        if graph_converter is None:
            from baseNet.graph.converters import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, lat, state_attr_default = graph_converter.get_graph(structure)
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_attr is None:
            state_attr = torch.tensor(state_attr_default)
        return self(g=g, state_attr=state_attr).detach()
