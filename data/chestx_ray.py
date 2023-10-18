import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ChestXray14Dataset(Dataset):
    '''
    Get image for train, validate and test base on NIH split
    '''

    def __init__(self, image_names, labels, transform, path, size, percentage=0.1):
        self.labels = labels
        self.percentage = percentage
        self.size = size
        self.image_names = image_names
        self.path = path
        self.transform = transform
        idx = np.random.permutation(len(self.labels))
        self.image_names, self.labels = self.image_names[idx], self.labels[idx]

    def __getitem__(self, index):
        image_file = self.path+self.image_names[index]
        image = Image.open(image_file).convert('RGB') # 1 channel image
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return int(self.image_names.shape[0] * self.percentage)

    @property
    def sz(self):
        # fastai compatible: learn.summary()
        return self.size
