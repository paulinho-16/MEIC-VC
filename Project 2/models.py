import torch
from torch import nn
from torchvision import models

from utils import Utils
from iterator import Iterator


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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