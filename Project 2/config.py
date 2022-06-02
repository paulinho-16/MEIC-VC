import torch

class Config:
    model_name = "vgg16"
    classes = ['trafficlight', 'stop', 'speedlimit', 'crosswalk']
    data_folder = './data'
    annotations_folder = './data/annotations/'
    images_folder = './data/images/'
    num_epochs = 10
    learning_rate = 0.05
    batch_size = 16
    num_workers = 2 
    device =  "cuda" if torch.cuda.is_available() else "cpu"