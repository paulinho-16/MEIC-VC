import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import torch

IMAGES_DIR = './data/'
ANNOTATIONS_DIR = './annotations/'
CLASSES = ['trafficlight', 'stop', 'speedlimit', 'crosswalk']

class TrafficSignsDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = pd.DataFrame(images, columns=['image_name'])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = io.imread(IMAGES_DIR + self.images.iloc[idx, 0] + '.png', as_gray=True)
        # image = cv2.imread(f'{IMAGES_DIR}{self.images.iloc[idx, 0]}.png')
        # image = np.array(image, dtype=np.uint8).reshape((280, 280))
        # image = Image.fromarray(image, mode='L')

        # image = Image.open(os.path.join(IMAGES_DIR, self.images.iloc[idx, 0] + '.png'))
        # image = image.resize((200, 200))
        # image = ImageOps.grayscale(image)

        image = cv2.imread(f'{IMAGES_DIR}{self.images.iloc[idx, 0]}.png')
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # TODO: COLOR_BGR2RGB?
        except:
            print(f'Error reading image {self.images.iloc[idx, 0]}.png')
            return None
        if self.transform:
            image = self.transform(image)

        tree = ET.parse(ANNOTATIONS_DIR + f'{self.images.iloc[idx, 0]}.xml')
        correct_labels = [movie.text for movie in tree.getroot().iter('name')]

        labels = []
        
        # TODO: Detect multiple classes, not just one
        for cl in CLASSES:
            labels.append(1) if cl in correct_labels else labels.append(0)

        teste = {'trafficlight': 0, 'stop': 1, 'speedlimit': 2, 'crosswalk': 3}

        #labels.append(1) if "speedlimit" in correct_labels else labels.append(0)
        if labels:
            labels = [teste[correct_labels[0]],]

        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype('long'))

        result = {
            'name': self.images.iloc[idx, 0],
            'image': image.float(),
            'labels': labels.float()
        }

        return result