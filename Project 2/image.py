import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from skimage import io
import xml.etree.ElementTree as ET
import torch

IMAGES_DIR = './data/'
ANNOTATIONS_DIR = './annotations/'
CLASSES = ['trafficlight', 'stop', 'speedlimit', 'crosswalk']

class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = pd.DataFrame(images, columns=['image_name'])
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(IMAGES_DIR + self.images.iloc[idx, 0] + '.png', as_gray=True)
        tree = ET.parse(ANNOTATIONS_DIR + f'{self.images.iloc[idx, 0]}.xml')
        correct_labels = [movie.text for movie in tree.getroot().iter('name')]

        labels = []
        for cl in CLASSES:
            labels.append(1) if cl in correct_labels else labels.append(0)

        if self.transform:
            image = self.transform(image)
        
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype('long'))

        return (image, labels)
