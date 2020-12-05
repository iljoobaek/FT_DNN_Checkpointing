import sys
import os
import torch

from torch.nn import ModuleList, Sequential, Conv2d, ReLU
from tag_layer import tag_all_pt_submodules

if __name__ == "__main__":
    print("Testing overriding of Pytorch layers")

    tag_all_pt_submodules(Sequential, True)

    extras = Sequential(ReLU())
    extras(torch.Tensor([5]))
    import pdb; pdb.set_trace()

