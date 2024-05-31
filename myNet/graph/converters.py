"""Tools to convert materials representations from Pymatgen and other codes to DGLGraphs."""

from __future__ import annotations

import abc

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from ase import Atoms
from pymatgen.core import Element, Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres

import myNet


def get_element_list(train_structures: list[Structure | Molecule]) -> tuple[str, ...]:
    """Get the tuple of elements in the training set for atomic features.

    Args:
        train_structures: Pymatgen Molecule/Structure object

    Returns:
        Tuple of elements covered in training set
    """
    elements: set[str] = set()
    for s in train_structures:
        elements.update(s.composition.get_el_amt_dict().keys())
    return tuple(sorted(elements, key=lambda el: Element(el).Z))


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """
        Args:
            structure: Input crystals or molecule

        Returns:
            DGLGraph object, state_attr
        """

    def get_graph_from_processed_structure(
            self,
            structure,
            src_id,
            dst_id,
            images,
            lattice_matrix,
            element_types,
            frac_coords,
    ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Construct a dgl graph from processed structure and bond information.

        Args:
            structure: Input crystals or molecule structure
            src_id: site indices for starting point of bonds
            dst_id: site indices for destination point of bonds
            images: Periodic image offsets for the bonds
            lattice_matrix: Lattice information of the structure
            element_types: Element symbols of all atoms in the structure
            frac_coords: Fractional coordinates of all atoms in the structure
                        (Note: Cartesian coordinates for molecule)

        Returns:
            DGLGraph object, lattice, state features
        """
        u, v = torch.tensor(src_id), torch.tensor(dst_id)
        g = dgl.graph((u, v), num_nodes=len(structure))
        pbc_offset = torch.tensor(images, dtype=myNet.float_th)
        g.edata["pbc_offset"] = pbc_offset
        lattice = torch.tensor(np.array(lattice_matrix), dtype=myNet.float_th)
        # Note: pbc_offshift and pos needs to be float64 to handle cases where bonds are exactly at cutoff
        element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
        if isinstance(structure, (Structure, Molecule)):
            node_type = np.array([element_types.index(site.specie.symbol) for site in structure])
        elif isinstance(structure, Atoms):
            node_type = np.array([element_to_index[elem] for elem in structure.get_chemical_symbols()])
        else:
            raise TypeError('Input must be pymatgen Structure/Molecule object, or ase Atoms object')
        g.ndata["node_type"] = torch.tensor(node_type, dtype=myNet.int_th)
        g.ndata["frac_coords"] = torch.tensor(frac_coords, dtype=myNet.float_th)
        state_attr = np.array([0.0, 0.0]).astype(myNet.float_np)
        return g, lattice, state_attr


class Molecule2Graph(GraphConverter):
    """Construct a DGL graph from Pymatgen Molecules."""

    def __init__(
            self,
            element_types: tuple[str, ...],
            cutoff: float = 5.0,
    ):
        """
        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                           constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, mol: Molecule) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Get a DGL graph from an input molecule.

        Args:
            mol: Pymatgen Molecule object

        Returns:
            DGLGraph object, lattice, state features

        """
        natoms = len(mol)
        R = mol.cart_coords
        element_types = self.element_types
        weight = mol.composition.weight / len(mol)
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        dists = mol.distance_matrix.flatten()
        nbonds = (np.count_nonzero(dists <= self.cutoff) - natoms) / 2
        nbonds /= natoms
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(natoms, dtype=np.bool_)
        adj = adj.tocoo()
        g, lat, _ = super().get_graph_from_processed_structure(
            mol,
            adj.row,
            adj.col,
            np.zeros((len(adj.row), 3)),
            np.expand_dims(np.identity(3), axis=0),
            element_types,
            R,
        )
        state_attr = [weight, nbonds]
        return g, lat, state_attr


class Structure2Graph(GraphConverter):
    """Construct a DGL graph from Pymatgen Structure."""

    def __init__(
            self,
            element_types: tuple[str, ...],
            cutoff: float = 5.0,
    ):
        """
        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                           constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, structure: Structure) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Get a DGL graph from an input Structure.

        Args:
            structure: Pymatgen Structure object

        Returns:
            DGLGraph object, lattice, state features
        """
        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=int)
        element_types = self.element_types
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, images, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
            bond_dist[exclude_self],
        )
        g, lat, state_attr = super().get_graph_from_processed_structure(
            structure,
            src_id,
            dst_id,
            images,
            [lattice_matrix],
            element_types,
            structure.frac_coords,
        )
        return g, lat, state_attr


class Atoms2Graph(GraphConverter):
    """Construct a DGL graph from ASE Atoms."""

    def __init__(
            self,
            element_types: tuple[str, ...],
            cutoff: float = 5.0,
    ):
        """Init Atoms2Graph from element types and cutoff radius.

        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                           constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, atoms: Atoms) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Get a DGL graph from an input Atoms.

        Args:
            atoms: Ase Atoms object

        Returns:
            DGLGraph object, lattice, state features
        """
        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=int)
        element_types = self.element_types
        lattice_matrix = np.array(atoms.get_cell()) if atoms.pbc.all() else np.expand_dims(np.identity(3), axis=0)
        cart_coords = atoms.get_positions()
        if atoms.pbc.all():
            src_id, dst_id, images, bond_dist = find_points_in_spheres(
                cart_coords,
                cart_coords,
                r=self.cutoff,
                pbc=pbc,
                lattice=lattice_matrix,
                tol=numerical_tol,
            )
            exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
            src_id, dst_id, images, bond_dist = (
                src_id[exclude_self],
                dst_id[exclude_self],
                images[exclude_self],
                bond_dist[exclude_self],
            )
        else:
            dist = np.linalg.norm(cart_coords[:, None, :] - cart_coords[None, :, :], axis=-1)
            adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(len(atoms.get_positions()), dtype=np.bool_)
            adj = adj.tocoo()
            src_id = adj.row
            dst_id = adj.col
        g, lat, state_attr = super().get_graph_from_processed_structure(
            atoms,
            src_id,
            dst_id,
            images if atoms.pbc.all() else np.zeros((len(adj.row), 3)),
            [lattice_matrix] if atoms.pbc.all() else lattice_matrix,
            element_types,
            atoms.get_scaled_positions(False) if atoms.pbc.all() else cart_coords,
        )

        return g, lat, state_attr
