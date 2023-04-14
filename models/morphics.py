import torch.nn as nn


class Morphics(nn.Module):
    def __init__(self, model):
        super(Morphics, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x
