import pandas as pd
import numpy as np
from pymatgen.core import Structure, Molecule
from tqdm import tqdm

import torch
from torch import nn
import pytorch_lightning as pl
import dgl
from dgl.data.utils import split_dataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from baseNet.models import MLPNet
from baseNet.graph.data import myDataset, myDataLoader, collate_fn
from baseNet.graph.converters import get_element_list, Molecule2Graph
from baseNet.utils.training import ModelLightningModule

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
# convert the raw dataset into baseNet Dataset
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

model = MLPNet([128, 1024, 100], dropout=0.05)
# print(model)
lit_module = ModelLightningModule(model=model)
logger = CSVLogger(".", name="logs")
checkpoint_callback = ModelCheckpoint(monitor='val_Total_Loss', save_last=True)
# early_stopping = EarlyStopping(monitor='val_Total_Loss', min_delta=0.0, patience=3, mode='min')
trainer = pl.Trainer(max_epochs=5,
                     accelerator="cuda",
                     logger=logger,
                     callbacks=[checkpoint_callback])
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

# prediction
pred_res = trainer.predict(model=lit_module, dataloaders=test_loader)
print(pred_res)
