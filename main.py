import torch
from torch import nn
from myNet.layers import MLP
from myNet.models import MLPNet
import myNet
# from matgl.ext.pymatgen import Molecule2Graph
import dgl

# test = MLP([128, 1024, 100])
# print(test)

test = MLPNet([128, 1024, 100])
print(test)

# print(myNet.int_th)
