from skimage import io
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

DATA = './data/'

class TrafficSignsDataset(Dataset):
  def __init__(self, images, labels, transform=None):
    self.images = pd.DataFrame (images, columns = ['image_name'])
    self.labels = labels
    self.transform = transform
  
  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(DATA +  self.images.iloc[idx, 0] + '.png')

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'landmarks': "Default"}


        return sample