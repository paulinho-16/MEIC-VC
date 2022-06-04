import cv2
import numpy as np
import pandas as pd
from bbox import BBox2D
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import torch

from config import Config

class ImageClassificationDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = pd.DataFrame(images, columns=['image_name'])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(f'{Config.images_folder}{self.images.iloc[idx, 0]}.png')
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(f'Error reading image {self.images.iloc[idx, 0]}.png')
            return None
        
        if self.transform:
            image = self.transform(image)

        tree = ET.parse(Config.annotations_folder + f'{self.images.iloc[idx, 0]}.xml')
        correct_labels = [movie.text for movie in tree.getroot().iter('name')]
        objects = [obj for obj in tree.getroot().iter('object')]
        objects = [(obj.find('name').text, [int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmax').text), int(obj.find('bndbox').find('ymax').text)]) for obj in objects]

        labels = []

        greater_area = 0
        label = None
        if correct_labels:
            for obj in objects:
                box = BBox2D(obj[1])
                area = box.height * box.width
                greater_area = area if area > greater_area else greater_area
                label = obj[0] if (area > greater_area or label is None) else label
        
        labels = Config.classes[label]
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype('long'))

        result = {
            'name': self.images.iloc[idx, 0],
            'image': image.float(),
            'labels': labels.float()
        }

        return result

class ImageMultiLabelDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = pd.DataFrame(images, columns=['image_name'])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(f'{Config.images_folder}{self.images.iloc[idx, 0]}.png')
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(f'Error reading image {self.images.iloc[idx, 0]}.png')
            return None
        if self.transform:
            image = self.transform(image)

        tree = ET.parse(Config.annotations_folder + f'{self.images.iloc[idx, 0]}.xml')
        correct_labels = [movie.text for movie in tree.getroot().iter('name')]

        labels = []
        for cl in Config.classes.keys():
            labels.append(1) if cl in correct_labels else labels.append(0)

        labels = np.asarray(labels)
        labels = torch.from_numpy(labels.astype('long'))

        result = {
            'name': self.images.iloc[idx, 0],
            'image': image.float(),
            'labels': labels.float()
        }

        return result