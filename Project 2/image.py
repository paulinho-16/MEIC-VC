from skimage import io
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

DATA = './data'

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

        print(idx)
        #img_name = os.path.join(DATA, self.images.iloc[idx, 0])

        image = io.imread('./data/road52.png')
        # landmarks = self.images.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = {'image': image, 'landmarks': "Default"}
        return sample