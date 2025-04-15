from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from dgl import softmax_nodes, sum_nodes

from baseNet import DEFAULT_ELEMENTS
from baseNet.layers import MLP, ActivationFunction, EmbeddingBlock

if TYPE_CHECKING:
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
            dropout: float = 0.0,
            element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
            cutoff: float = 4.0,
            **kwargs):
        """

        Args:
            dims: Dimensions of each layer of MLP
            dim_node_embedding: Dimension of node embedding
            activation_type: Activation used for non-linearity
            activate_last: Whether to apply activation to last layer
            bias_last: Whether to apply bias to last layer
            n_layers: Number of layers in MLP
            nlayers_set2set: Number of layers in Set2Set layer
            niters_set2set: Number of iterations in Set2Set layer
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
                a Bernoulli distribution. Defaults to 0, i.e., no dropout.
            element_types: Elements included in the training set
            cutoff: cutoff for forming bonds
            **kwargs: For future flexibility. Not used at the moment.
        """
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
        self.out = MLP([dims[-1], 1])
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(
            self,
            g: dgl.DGLGraph,
            state_attr: torch.Tensor | None = None,
            **kwargs,
    ):
        """
        Args:
            g: dgl Graph
            state_attr: State attribute
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs
        """
        node_attr = g.ndata["node_type"]
        node_feat = self.embedding(node_attr)
        for block in self.MLPblock:
            node_feat = block(node_feat)

        if self.dropout:
            node_feat = self.dropout(node_feat)
        node_feat = self.out(node_feat)
        g.ndata["e"] = node_feat
        alpha = softmax_nodes(g, "e")
        g.ndata["r"] = node_feat * alpha
        output = sum_nodes(g, "r")

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
