import numpy as np
# import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from train import Train
from image import TrafficSignsDataset
from neural_network import ConvolutionalNeuralNetwork

def read_images(filename):
    images = []
    with open(filename) as file:
        while (line := file.readline().rstrip()):
            images.append(line)
    return images

def get_mean_and_std(loader):
    mean = 0
    std = 0
    total_images = 0
    for batch, _ in loader:
        batch_size = batch.size(0)
        print(batch.shape)
        batch = batch.view(batch_size, batch.size(1), -1)
        print(batch.shape)

transform = transforms.Compose([ # TODO: try another values/transformations
    # transforms.ToPILImage(),
    # transforms.Resize((100, 100)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

train_images = read_images('train.txt')
test_images = read_images('test.txt')

train_dataset = TrafficSignsDataset(train_images, None, transform)
test_dataset = TrafficSignsDataset(test_images, None, transform)

# divide dataset into train-val-test subsets
indices = list(range(len(test_dataset)))
np.random.shuffle(indices, )

test_size = 0.2 * len(indices)
split = int(np.floor(test_size))
val_idx, test_idx = indices[split:], indices[:split]

validation_dataset = SubsetRandomSampler(val_idx)
test_dataset = SubsetRandomSampler(test_idx)

print(f'Training size: {len(train_dataset)}\nValidation size: {len(val_idx)} \nTest size: {len(test_idx)}')

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
validation_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True, drop_last=False)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)

# get_mean_and_std(train_dl)

model = ConvolutionalNeuralNetwork().to(device) # put model in device (GPU or CPU)
print(model)

output = Train(device, model, train_dl, validation_dl, test_dl)