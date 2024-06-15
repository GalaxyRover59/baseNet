from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from dgl.nn import Set2Set

from baseNet import DEFAULT_ELEMENTS
from baseNet.layers import MLP, ActivationFunction, EmbeddingBlock

# if TYPE_CHECKING:
import dgl
from baseNet.graph.converters import GraphConverter


class MLPNet(nn.Module):
    def __init__(
            self,
            dims,
            dim_node_embedding: int = 16,
            activation_type: str = "softplus",
            activate_last: bool = False,
            bias_last: bool = True,
            n_layers: int = 3,
            nlayers_set2set: int = 1,
            niters_set2set: int = 2,
            dropout: float = 0.0,
            element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
            cutoff: float = 4.0,
            **kwargs):
        super().__init__()
        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.embedding = EmbeddingBlock(
            dim_node_embedding=dim_node_embedding,
            ntypes_node=len(self.element_types),
            activation=activation,
        )
        self.MLPblock = nn.ModuleList(
            [MLP([dim_node_embedding] + dims, activation, activate_last, bias_last)] + [
                MLP([dims[-1]] + dims, activation, activate_last, bias_last)
                for _ in range(n_layers - 1)
            ]
        )
        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.node_s2s = Set2Set(dims[-1], **s2s_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.out = MLP([2 * dims[-1], 1])

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
        node_attr = g.ndata["node_type"]
        node_feat = self.embedding(node_attr)
        for block in self.MLPblock:
            node_feat = block(node_feat)

        node_vec = self.node_s2s(g, node_feat)
        if self.dropout:
            node_vec = self.dropout(node_vec)
        output = self.out(node_vec)

        return torch.squeeze(output)

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
