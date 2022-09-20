import sys
import torch
from torch import nn
from torchvision import transforms
import gc

gc.collect()

torch.cuda.empty_cache()

###################################################
# Global Variables
###################################################
from config import Config

print(f"Using {Config.device} device\n")

###################################################
# Transforms
###################################################
transforms_dict = {
    "train": transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((Config.images_size, Config.images_size)), 
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor()
                ]),
    "validation": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((Config.images_size, Config.images_size)), 
                    transforms.ToTensor()
                ]),
    "test": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((Config.images_size, Config.images_size)),
                transforms.ToTensor()
            ])
}

###################################################
# Read Images
###################################################
from dataset import ImageClassificationDataset, ImageMultiLabelDataset

def read_images(filename):
    images = []
    with open(filename) as file:
        while (line := file.readline().rstrip()):
            images.append(line)
    return images

val_train_images = read_images('./data/train.txt')
test_images = read_images('./data/test.txt')

train_ratio = int(0.8 * len(val_train_images))
validation_ratio = len(val_train_images) - train_ratio

train_images = list(val_train_images[:train_ratio])
validation_images = list(val_train_images[-validation_ratio:])

###################################################
# Models
###################################################
from models import ClassificationVGG16, ClassificationResNet, ClassificationCustomModel, ClassificationMultilabel

if __name__ == "__main__":
    version = input('Enter the desired version (basic, intermediate, advanced): ')

    if version == 'basic':
        architecture = input('Choose the architecture (vgg16, resnet): ')
        if architecture == 'vgg16':
            neural_network = ClassificationVGG16(True)
        elif architecture == 'resnet':
            neural_network = ClassificationResNet(True)
        else:
            sys.exit('Invalid architecture')
    elif version == 'intermediate':
        neural_network = ClassificationCustomModel(True)
    elif version == 'advanced':
        model = input('Choose the model (vgg16, resnet, custom): ')
        neural_network = ClassificationMultilabel(model, True)
    else:
        sys.exit('Invalid version')

    if version == 'advanced':
        train_data = ImageMultiLabelDataset(train_images, transforms_dict['train'])
        validation_data = ImageMultiLabelDataset(validation_images, transforms_dict['validation'])
        test_data = ImageMultiLabelDataset(test_images, transforms_dict['test'])
    else:
        train_data = ImageClassificationDataset(train_images, transforms_dict['train'])
        validation_data = ImageClassificationDataset(validation_images, transforms_dict['validation'])
        test_data = ImageClassificationDataset(test_images, transforms_dict['test'])

    print(f'Training size: {len(train_data)}\nValidation size: {len(validation_data)} \nTest size: {len(test_data)}\n')

    labels_quantity = {'trafficlight': [], 'stop': [], 'speedlimit': [], 'crosswalk': []}
    for image in train_data:
        if version == 'advanced':
            labels_idx = [i for i, x in enumerate(list(image['labels'])) if x == 1]
            labels = [list(labels_quantity)[idx] for idx in labels_idx]
            for label in labels:
                labels_quantity[label].append(image)
        else:
            label = list(labels_quantity)[int(image['labels'].item())]
            labels_quantity[label].append(image)

    print('Labels quantity:')
    for key, value in labels_quantity.items():
        print(f'\t{key}: {len(value)} images')

    total_presences = sum([len(value) for value in labels_quantity.values()])
    weights = [1 - len(value)/total_presences for value in labels_quantity.values()]

    if version == 'advanced':
        loss_fn = nn.BCELoss(weight = torch.tensor(weights, dtype=torch.float, device=Config.device))
    else:
        loss_fn = nn.CrossEntropyLoss(weight = torch.tensor(weights, dtype=torch.float, device=Config.device))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    validation_data = torch.utils.data.DataLoader(validation_data, batch_size=Config.batch_size, shuffle=False, drop_last=False)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)
    
    neural_network.run(train_data, test_data, validation_data, loss_fn)
