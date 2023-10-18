import torch.nn as nn
import pretrainedmodels
from torchvision.models import densenet121
import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

import pdb
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class ChexNet(nn.Module):
    tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])

    def __init__(self, trained=False, path=Path('snapshots_models')):
        super().__init__()
        # chexnet.parameters() is freezed except head
        if trained:
            self.load_model(path)
        else:
            self.load_pretrained()
        self.sft = nn.Softmax(dim=1)

    def load_model(self, path):
        self.backbone = densenet121(False).features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 Flatten(),
                                 nn.Linear(1024, 6))
        state_dict = torch.load(path/'chexnet.h5')
        state_dict['head.2.weight'] = state_dict['head.2.weight'][[0,2,3,4,5,7],:]
        state_dict['head.2.bias'] = state_dict['head.2.bias'][[0,2,3,4,5,7]]
        self.load_state_dict(state_dict)

    def load_pretrained(self, torch=False):
        if torch:
            # torch vision, train the same -> ~0.75 AUC on test
            self.backbone = densenet121(True).features
        else:
            # pretrainmodel, train -> 0.85 AUC on test
            self.backbone = pretrainedmodels.__dict__['densenet121']().features

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  Flatten(),
                                  nn.Linear(1024, 6))

    def forward(self, x):
        output = self.head(self.backbone(x)).view(-1, 10, 6).mean(1)
        return output