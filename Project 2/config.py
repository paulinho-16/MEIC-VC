import torch

class Config:
    model_name = "vgg16"
    classes = {'trafficlight': 0, 'stop': 1, 'speedlimit': 2, 'crosswalk': 3}
    data_folder = './data'
    annotations_folder = './data/annotations/'
    images_folder = './data/images/'
    images_size = 300
    num_epochs = 100
    learning_rate = 0.05
    batch_size = 16
    num_filters = 32
    kernel_size = 5
    pool_size = 2
    padding = 0
    stride = 1
    num_workers = 2 
    device =  "cuda" if torch.cuda.is_available() else "cpu"