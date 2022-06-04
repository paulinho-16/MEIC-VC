import sys
from torch import nn
from math import prod
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

    def model(self):
        model = models.resnet50(pretrained=self.pre_trained)
        model.fc = nn.Linear(2048, 4)
        model.to(Config.device)
        return model

    def run(self, train_dl, test_dl, validation_dl):
        model = self.model()

        loss_function = nn.CrossEntropyLoss() 

        train_history, val_history = Iterator.train(model, train_dl, validation_dl, loss_function)
        Utils.learning_curve_graph(train_history, val_history)

        Iterator.test(model, test_dl, loss_function)

class ClassificationVGG16:
    def __init__(self, pre_trained = True):
        self.pre_trained = pre_trained

    def model(self):
        model = models.vgg16(pretrained=self.pre_trained)
        model.classifier[6] = nn.Linear(4096, 4)
        model.to(Config.device)
        return model

    def run(self, train_dl, test_dl, validation_dl):
        model = self.model()

        loss_function = nn.BCELoss()

        train_history, val_history = Iterator.train(model, train_dl, validation_dl, loss_function)
        Utils.learning_curve_graph(train_history, val_history)

        Iterator.test(model, test_dl, loss_function)

###################################################
# Intermediate
###################################################
class ClassificationCustomNetwork(nn.Module):
    def __init__(self):
        super(ClassificationCustomNetwork, self).__init__()
        self.num_conv_layer = 2
        self.num_max_pool = 1
        output_size = Config.images_size
        for _ in range(self.num_conv_layer):
            output_size = Utils.calculate_output_size(output_size)
        self.output_shape = (output_size, output_size, Config.num_filters)

        for _ in range(self.num_max_pool):
            self.output_shape = (self.output_shape[0]/Config.pool_size, self.output_shape[1]/Config.pool_size, self.output_shape[2])

        self.layers = nn.Sequential(
            nn.Conv2d(3, Config.num_filters, Config.kernel_size),
            nn.ReLU(),
            nn.Conv2d(Config.num_filters, Config.num_filters, Config.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(Config.pool_size),

            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(int(prod(self.output_shape)), 128),
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

    def model(self):
        return ClassificationCustomNetwork().to(Config.device)

    def run(self, train_dl, test_dl, validation_dl):
        model = self.model()

        loss_function = nn.CrossEntropyLoss()

        train_history, val_history = Iterator.train(model, train_dl, validation_dl, loss_function)
        Utils.learning_curve_graph(train_history, val_history)

        Iterator.test(model, test_dl, loss_function)

###################################################
# Advanced - adapt the previous models to solve the original problem, i.e. multilabel classification, and compare their performance
###################################################
class ClassificationMultilabel:
    def __init__(self, model_name, pre_trained = True):
        self.model_name = model_name
        self.pre_trained = pre_trained

    def model(self):
        if self.model_name == 'vgg16': return ClassificationVGG16().model()
        elif self.model_name == 'resnet': return ClassificationResNet().model()
        elif self.model_name == 'custom': return ClassificationCustomModel().model()
        sys.exit('Invalid model')

    def run(self, train_dl, test_dl, validation_dl): 
        model = ClassificationCustomNetwork().to(Config.device)

        loss_function = nn.CrossEntropyLoss()

        train_history, val_history = Iterator.train(model, train_dl, validation_dl, loss_function, multilabel=True)
        Utils.learning_curve_graph(train_history, val_history)

        Iterator.test(model, test_dl, loss_function, multilabel=True)