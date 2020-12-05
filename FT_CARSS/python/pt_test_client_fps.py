from tag_layer_fps import tag_pt_module_layers_at_depth, FrameController
from torch.nn import Sequential, Conv2d, ReLU, Module
import torch

class TestSSD(Module):
    def __init__(self, *args, **kwargs):
        super(TestSSD, self).__init__(*args, **kwargs)

        self.layer1 = Sequential(ReLU(), ReLU())
        self.layer2 = Sequential(ReLU(), ReLU())
        self.add_module("Layer1", self.layer1)
        self.add_module("Layer2", self.layer2)
        self.name = "TestSSD"

    def forward(self, x):
        print("In outer forward!")
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    
if __name__ == "__main__":
    print("Testing explicit overriding of an instance's layers")

    fc = FrameController("TestSSD forward", 1.0)
    ssd = TestSSD()
    tag_pt_module_layers_at_depth(ssd, fc, True, -1)

    for i in range(20):
        print("Starting frame!")
        fc.frame_start()
        ssd(torch.Tensor([5]))
        fc.frame_end()
    

