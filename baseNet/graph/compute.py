"""Computing various graph based operations."""

from __future__ import annotations

from typing import Callable

import dgl
import numpy as np
import torch

import baseNet


def compute_pair_vector_and_distance(g: dgl.DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate bond vectors and distances using dgl graphs.

    Args:
        g: dgl Graph

    Returns:
        bond distance, vector from src node to dst node
    """
    dst_pos = g.ndata["pos"][g.edges()[1].type(torch.long)] + g.edata["pbc_offshift"]
    src_pos = g.ndata["pos"][g.edges()[0].type(torch.long)]
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)
    return bond_vec, bond_dist
