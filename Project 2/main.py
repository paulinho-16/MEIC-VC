"""
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(42)

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image

from image import TrafficSignsDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

DATA = './data'
BATCH_SIZE = 20
NUM_WORKERS = 2

# get cpu or gpu device for training

traffic_sign_dataset = TrafficSignsDataset(ims=X_train)
train_dl = DataLoader(traffic_sign_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

print(train_dl[0])

"""

# Vou comentar tudo, vamos fazer direitinho coisa por coisa

import os
print(os.listdir("./data"))

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from image import TrafficSignsDataset

def read_images():
    train_images = []
    with open('train.txt') as file:
        while (line := file.readline().rstrip()):
            train_images.append(line)

transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])



trafficSignsDataset = TrafficSignsDataset(train_images, None, transform)

trainset = torch.utils.data.DataLoader(trafficSignsDataset, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(trafficSignsDataset, batch_size=10, shuffle=True)

for batch in trainset:
    print("Data: ", batch)
    break