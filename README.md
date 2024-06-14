# baseNet
A graph neural network project that only includes basic functions in PyTorch. Using Deep Graph Library(dgl) package.

    ### DGL

    #### •Installation
    See [Install from Conda or Pip](https://www.dgl.ai/pages/start.html).

    #### •User guide
    See [User Guide](https://docs.dgl.ai/en/1.1.x/guide/index.html). And the chineses version is [用户指南](https://docs.dgl.ai/en/1.1.x/guide_cn/index.html).

## data
This package implements data manipulation tools.

Include normalizer and log transformer currently.

## ext
This package implements interfaces to external packages such as Pymatgen and the Atomic Simulation Environment.

## graph
Package for creating and manipulating graphs.

    ### converters
    Tools to convert materials representations from external packages to DGLGraphs.
    
    Currently convertible instances:
    
    • pymatgen: Structure, Molecule
    • ASE: Atoms
    
    ### data
    Tools to construct a dataset of DGL graphs.
    
    ### compute
    Computing various graph based operations.

## layers
This package implements the layers for baseNet.

## models
Package containing model implementations.

## utils
Implementation of various utility methods and classes.
