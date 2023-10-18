from torch import nn
import torch


class CDCNet(nn.Module):
    def __init__(self, path=None):
        super(CDCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(54, 14),
            nn.ReLU(),
            nn.Linear(14, 2)
        )

        if path != None:
            self.network.load_state_dict(torch.load(path))
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)

        return x