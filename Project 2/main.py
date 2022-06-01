from __future__ import annotations
import torch
from torchvision import transforms

###################################################
# Global Variables
###################################################
class Config:
    model_name = "vgg16"
    data_folder = './data'
    annotations_folder = '/annotations'
    images_folder = '/images'
    num_epochs = 2
    learning_rate = 0.05
    batch_size = 16
    num_workers = 2 
    device =  "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {Config.device} device\n")
###################################################
# Transforms
###################################################
transforms_dict = {
    "train": transforms.Compose([ 
                    transforms.ToPILImage(),
                    transforms.Resize((200, 200)), 
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor()
                ]),
    "validation": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((200, 200)), 
                    transforms.ToTensor()
                ]),
    "test": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.ToTensor()
            ])
}

###################################################
# Read Images
###################################################
from dataset import ImageClassificationDataset

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

train_data =ImageClassificationDataset(train_images, transforms_dict['train'])
validation_data = ImageClassificationDataset(validation_images, transforms_dict['validation'])
test_data = ImageClassificationDataset(test_images, transforms_dict['test'])

train_data = torch.utils.data.DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
validation_data = torch.utils.data.DataLoader(validation_data, batch_size=Config.batch_size, shuffle=False, drop_last=False)
test_data = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)
print(f'Training size: {len(train_data)}\nValidation size: {len(validation_data)} \nTest size: {len(test_data)}\n')

###################################################
# Models
###################################################
from models import ClassificationVGG16, ClassificationResNet

if __name__ == "__main__":
    neural_network = ClassificationVGG16(True)
    neural_network = ClassificationResNet(True)

    neural_network.run(train_data, test_data, validation_data)

    """version = input('Enter the desired version (basic, intermediate, advanced): ')

    if version == 'basic':
        architecture = input('Choose the architecture (vgg16, resnet): ')
        model, loss_fn, optimizer = BasicVersion(architecture).get_model()
        model = model.to(device)
    else: # TODO: other versions
        model = None

    Train(device, model, loss_fn, optimizer, NUM_EPOCHS, train_dl, validation_dl, test_dl)"""
