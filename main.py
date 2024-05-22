import pandas as pd
import numpy as np
from pymatgen.core import Structure, Molecule
from tqdm import tqdm

import torch
from torch import nn
import dgl
from dgl.data.utils import split_dataset

from myNet.layers import MLP
from myNet.models import MLPNet
from myNet.graph.data import myDataset, myDataLoader, collate_fn
from myNet.graph.converters import get_element_list, Molecule2Graph

data = pd.read_csv("qm9_sample.csv")


def load_dataset(dataframe) -> tuple[list[Structure], list[str], list[float]]:
    structures = []
    mol_ids = []
    energy = []
    stress = []

    for i in tqdm(range(len(dataframe))):
        mol = Molecule.from_dict(eval(dataframe["structure"][i]))

        eles = [mol[i].species_string for i in range(len(mol))]
        coords = mol.cart_coords.astype('float32')
        mol = Molecule(eles, coords)

        structures.append(mol)
        mol_ids.append(dataframe["struct_id"][i])
        energy.append(float(dataframe["energy"][i]))
        stress.append(np.zeros((3, 3)).tolist())

    return structures, mol_ids, energy, stress


molecules, mol_ids, energy, stress = load_dataset(data)

elem_list = get_element_list(molecules)
# setup a graph converter
converter = Molecule2Graph(element_types=elem_list, cutoff=4.0)
# convert the raw dataset into MEGNetDataset
mp_dataset = myDataset(
    structures=molecules,
    labels={
        "energies": energy,
    },
    converter=converter,
)

train_data, val_data, test_data = split_dataset(
    mp_dataset,
    frac_list=[0.9, 0.05, 0.05],
    shuffle=True,
    random_state=42,
)

train_loader, val_loader, test_loader = myDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn,
    batch_size=16,
    num_workers=0,
)

for batch in train_loader:
    g, lat, state_attr, labels = batch
    # print(g)
    break

model = MLPNet([128, 1024, 100], dropout=0.05)
out = model(g)
print(f'length: {len(out)}')
print(out)

# print(myNet.int_th)
