import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from utils import Utils
from iterator import Iterator


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

###################################################
# Basic
###################################################
class ClassificationResNet:
    def __init__(self, pre_trained = True):
        self.pre_trained = pre_trained

    def run(self, train_dl, test_dl, validation_dl):
        model = models.resnet50(pretrained=self.pre_trained)
        model.fc = nn.Linear(2048, 4)

        model.to(device)

        loss_function = nn.CrossEntropyLoss() 

        train_history, val_history = Iterator.train(model, train_dl, validation_dl, loss_function)
        Utils.learning_curve_graph(train_history, val_history)

        Iterator.test(model, test_dl, loss_function)

class ClassificationVGG16:
    def __init__(self, pre_trained = True):
        self.pre_trained = pre_trained

    def run(self, train_dl, test_dl, validation_dl):
        model = models.vgg16(pretrained=self.pre_trained)

        model.classifier[6] = nn.Linear(4096, 4)
        model.to(device)

        loss_function = nn.CrossEntropyLoss()

        train_history, val_history = Iterator.train(model, train_dl, validation_dl, loss_function)
        Utils.learning_curve_graph(train_history, val_history)

        Iterator.test(model, test_dl, loss_function)

###################################################
# Intermediate
###################################################
class ClassificationCustomNetwork(nn.Module):
    def __init__(self):
        super(ClassificationCustomNetwork, self).__init__()
        self.pool_size = 2
        self.nb_filters = 32
        self.kernel_size = 3

        self.layers = nn.Sequential(
            nn.Conv2d(1, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.nb_filters, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),

            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(307328, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

        self.conv1 = nn.Conv2d(3, self.nb_filters, self.kernel_size)
        self.pool = nn.MaxPool2d(self.pool_size)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, self.kernel_size)
        self.fc1 = nn.Linear(self.nb_filters * 48 * 48, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

###################################################
# Advanced - adapt the previous models to solve the original problem, i.e. multilabel classification, and compare their performance
###################################################
