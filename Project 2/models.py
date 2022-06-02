from torch import nn
from torchvision import models

from utils import Utils
from iterator import Iterator
from config import Config

###################################################
# Basic
###################################################
class ClassificationResNet:
    def __init__(self, pre_trained = True):
        self.pre_trained = pre_trained

    def run(self, train_dl, test_dl, validation_dl):
        model = models.resnet50(pretrained=self.pre_trained)
        model.fc = nn.Linear(2048, 4)

        model.to(Config.device)

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
        model.to(Config.device)

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
        self.kernel_size = 5

        self.layers = nn.Sequential(
            nn.Conv2d(3, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.nb_filters, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),

            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(294912, 128), #TODO: Como obter este valor? Quais são as contas?
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

class ClassificationCustomModel:
    def __init__(self, pre_trained = True):
        self.pre_trained = pre_trained

    def run(self, train_dl, test_dl, validation_dl):
        model = ClassificationCustomNetwork().to(Config.device) 

        loss_function = nn.CrossEntropyLoss()

        train_history, val_history = Iterator.train(model, train_dl, validation_dl, loss_function)
        Utils.learning_curve_graph(train_history, val_history)

        Iterator.test(model, test_dl, loss_function)
###################################################
# Advanced - adapt the previous models to solve the original problem, i.e. multilabel classification, and compare their performance
###################################################
