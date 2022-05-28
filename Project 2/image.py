from skimage import io
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import torch

IMAGES_DIR = './data/'
ANNOTATIONS_DIR = './annotations/'

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

        image = io.imread(IMAGES_DIR +  self.images.iloc[idx, 0] + '.png')
        tree = ET.parse(ANNOTATIONS_DIR + f'{self.images.iloc[idx, 0]}.xml')

        labels = [movie.text for movie in tree.getroot().iter('name')]
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'labels': {
            'stop': 1 if 'stop' in labels else 0,
            'trafficlight': 1 if 'trafficlight' in labels else 0,
            'speedlimit': 1 if 'speedlimit' in labels else 0,
            'crosswalk': 1 if 'crosswalk' in labels else 0
        }
        }

        return sample