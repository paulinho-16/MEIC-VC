import numpy as np
# import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from train import Train
from dataset import TrafficSignsDataset
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

train_transform = transforms.Compose([ # TODO: try another values/transformations
    transforms.ToPILImage(),
    transforms.Resize((200, 200)), # (400, 400)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

validation_transform = transforms.Compose([ # TODO: try another values/transformations
    transforms.ToPILImage(),
    transforms.Resize((200, 200)), # (400, 400)
    transforms.ToTensor()
])

test_transform = transforms.Compose([ # TODO: try another values/transformations
    transforms.ToPILImage(),
    transforms.Resize((200, 200)), # TODO: acho que não devia ter resize (ver link)
    transforms.ToTensor()
])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

val_train_images = read_images('train.txt')
test_images = read_images('test.txt')

train_ratio = int(0.8 * len(val_train_images))
validation_ratio = len(val_train_images) - train_ratio

train_images = list(val_train_images[:train_ratio])
validation_images = list(val_train_images[-validation_ratio:])

train_data = TrafficSignsDataset(train_images, train_transform)
validation_data = TrafficSignsDataset(validation_images, validation_transform)
test_data = TrafficSignsDataset(test_images, test_transform)

# # divide dataset into train-val-test subsets
# indices = list(range(len(test_dataset)))
# np.random.shuffle(indices, )

# test_size = 0.2 * len(indices)
# split = int(np.floor(test_size))
# val_idx, test_idx = indices[split:], indices[:split]

# validation_dataset = SubsetRandomSampler(val_idx)
# test_dataset = SubsetRandomSampler(test_idx)

print(f'Training size: {len(train_data)}\nValidation size: {len(validation_data)} \nTest size: {len(test_data)}')

train_dl = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
validation_dl = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=False, drop_last=False)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False)

# get_mean_and_std(train_dl)

model = ConvolutionalNeuralNetwork().to(device) # put model in device (GPU or CPU)
# print(model)

output = Train(device, model, train_dl, validation_dl, test_dl)