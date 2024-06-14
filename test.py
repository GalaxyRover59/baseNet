import numpy as np
import torch as th
import dgl

u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g1 = dgl.graph((u, v))
g2 = dgl.graph((v, u))

g_batch = [g1, g2]
g1.ndata['number'] = th.tensor([2.1, 424.2, 114.9, 514.2])
print(g1.ndata)