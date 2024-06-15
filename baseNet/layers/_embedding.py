"""Embedding node, edge and optional state attributes."""

from __future__ import annotations

from torch import nn

import baseNet


class EmbeddingBlock(nn.Module):
    """Embedding block for generating node, bond and state features."""

    def __init__(
            self,
            activation: nn.Module,
            dim_node_embedding: int,
            ntypes_node: int | None = None,
    ):
        """
        Args:
            activation: Activation type
            dim_node_embedding: Dimensionality of node features
            ntypes_node: Number of node labels
        """
        super().__init__()
        self.dim_node_embedding = dim_node_embedding
        self.ntypes_node = ntypes_node
        self.activation = activation

        if ntypes_node is not None:
            self.layer_node_embedding = nn.Embedding(ntypes_node, dim_node_embedding)
        else:
            self.layer_node_embedding = nn.Sequential(
                nn.LazyLinear(dim_node_embedding, bias=False, dtype=baseNet.float_th),
                activation,
            )

    def forward(self, node_attr):
        """Output embedded features.

        Args:
            node_attr: node attribute

        Returns:
            node_feat: embedded node features
        """
        if self.ntypes_node is not None:
            node_feat = self.layer_node_embedding(node_attr)
        else:
            node_feat = self.layer_node_embedding(node_attr.to(baseNet.float_th))

        return node_feat
