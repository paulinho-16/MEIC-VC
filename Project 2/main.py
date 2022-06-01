from pickletools import optimize
import numpy as np
import torch
from torchvision import transforms

from train import Train
from models import BasicVersion
from dataset import ImageClassificationDataset
from neural_network import ConvolutionalNeuralNetwork

BATCH_SIZE = 16
NUM_EPOCHS = 10

def read_images(filename):
    images = []
    with open(filename) as file:
        while (line := file.readline().rstrip()):
            images.append(line)
    return images

def get_mean_and_std(loader): # TODO: Normalize images (on train)
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

train_data = ImageClassificationDataset(train_images, train_transform)
validation_data = ImageClassificationDataset(validation_images, validation_transform)
test_data = ImageClassificationDataset(test_images, test_transform)

print(f'Training size: {len(train_data)}\nValidation size: {len(validation_data)} \nTest size: {len(test_data)}\n')

train_dl = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_dl = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# get_mean_and_std(train_dl)

if __name__ == "__main__":
    version = input('Enter the desired version (basic, intermediate, advanced): ')

    if version == 'basic':
        architecture = input('Choose the architecture (vgg16, resnet): ')
        model, loss_fn, optimizer = BasicVersion(architecture).get_model()
        model = model.to(device)
    else: # TODO: other versions
        model = None

    Train(device, model, loss_fn, optimizer, NUM_EPOCHS, train_dl, validation_dl, test_dl)
