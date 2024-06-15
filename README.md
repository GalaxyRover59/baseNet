# baseNet

A graph neural network project that only includes basic functions in PyTorch, using Deep Graph Library(dgl) package.

And this project draws inspiration from the [Materials Graph Library](https://github.com/materialsvirtuallab/matgl).

### ◇ DGL

#### • Installation

See [Install from Conda or Pip](https://www.dgl.ai/pages/start.html).

#### • User guide

See [User Guide](https://docs.dgl.ai/en/1.1.x/guide/index.html). The chineses version
is [用户指南](https://docs.dgl.ai/en/1.1.x/guide_cn/index.html).

---

## ● data

This package implements data manipulation tools.

Include normalizer and log transformer currently.

## ● ext

This package implements interfaces to external packages such as Pymatgen and the Atomic Simulation Environment.

## ● graph

Package for creating and manipulating graphs.

### ◇ compute

Computing various graph based operations.

### ◇ converters

Tools to convert materials representations from external packages to DGLGraphs.

Currently convertible instances:

• [pymatgen](https://pymatgen.org/pymatgen.html): Structure, Molecule

• [ASE](https://wiki.fysik.dtu.dk/ase/ase/ase.html): Atoms

### ◇ data

Tools to construct a dataset of DGL graphs.

## ● layers

This package implements the layers for baseNet.

### ◇ activations

Integrate some activation functions.

### ◇ core

Implementations of multi-layer perceptron (MLP) and other helper classes.

### ◇ embedding

Embedding node.

## ● models

Package containing model implementations.

As an example, build a model with MLP as the main body, called MLPNet.

## ● utils

Implementation of various utility methods and classes.

### ◇ training

Utils for training baseNet models.

---

# qm9_sample.csv

128 samples
from [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904)
dataset.

struct_id | structure | energy
---|---|---
id in QM9 dataset|pymatgen Molecule| 0K energy (Hartree)