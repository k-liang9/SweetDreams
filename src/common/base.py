import torch.nn as nn

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, x, target=None):
        raise NotImplementedError()
    
    def featurize(self, batch):
        raise NotImplementedError()
    
    def compute_metrics(self, split, out, x, target):
        raise NotImplementedError()
