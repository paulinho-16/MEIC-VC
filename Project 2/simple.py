import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import TrafficSignsDataset

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200*200, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def read_images(filename):
    images = []
    with open(filename) as file:
        while (line := file.readline().rstrip()):
            images.append(line)
    return images
    
def plotTrainingHistory(train_history, val_history):
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(train_history['loss'], label='train')
    plt.plot(val_history['loss'], label='val')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    plt.plot(train_history['accuracy'], label='train')
    plt.plot(val_history['accuracy'], label='val')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

def epoch_iter(dataloader, model, loss_fn, optimizer=None, is_train=True):
    if is_train:
        assert optimizer is not None, "When training, please provide an optimizer."
    
    num_batches = len(dataloader)

    if is_train:
        model.train() # put model in train mode
    else:
        model.eval()

    total_loss = 0.0
    preds = []
    labels = []

    with torch.set_grad_enabled(is_train):
        for batch, data in enumerate(tqdm(dataloader)):
            inputs, label = data['image'], data['labels']

            # Expected all tensors to be on the same device
            label = label.type(torch.LongTensor) 
            inputs, label = inputs.to(device), label.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, label)
            #print(outputs)
            #print(label)

            if is_train:
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            
            # Save training metrics
            total_loss += loss.item() # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached
            
            probs = F.softmax(outputs, dim=1)
            final_pred = torch.argmax(probs, dim=1)
            preds.extend(final_pred.cpu().numpy())
            
            labels.extend(label.cpu().numpy())

    return total_loss / num_batches, accuracy_score(labels, preds)

def showErrors(model, dataloader, num_examples=20):    
    plt.figure(figsize=(15, 15))

    for ind, data in enumerate(tqdm(dataloader)):
        if ind >= 90: break
        inputs, label = data['image'], data['labels']

        # Expected all tensors to be on the same device
        label = label.type(torch.LongTensor) 
        inputs, label = inputs.to(device), label.to(device)
        
        pred = model(inputs)
        probs = F.softmax(pred, dim=1)
        final_pred = torch.argmax(probs, dim=1)

        plt.subplot(10, 10, ind + 1)
        plt.axis("off")
        plt.text(0, -1, label[0].item(), fontsize=14, color='green') # correct
        plt.text(8, -1, final_pred[0].item(), fontsize=14, color='red')  # predicted
        plt.imshow(inputs[0].cpu(), cmap='gray')
    plt.show()

############################################################################################################
############################################################################################################
############################################################################################################
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


if __name__ == "__main__":
    # now we need to define a Dataloader, which allows us to automatically batch our inputs, do sampling and multiprocess data loading
    batch_size = 64
    num_workers = 2 # how many processes are used to load the data

    train_images = read_images('train.txt')
    val_test_images = read_images('test.txt')

    test_ratio = int(0.8 * len(val_test_images))
    validation_ratio = len(val_test_images) - test_ratio

    test_images = list(val_test_images[:test_ratio])
    validation_images = list(val_test_images[-validation_ratio:])

    train_dataloader = TrafficSignsDataset(train_images, train_transform)
    validation_dataloader = TrafficSignsDataset(validation_images, validation_transform)
    test_dataloader = TrafficSignsDataset(test_images, test_transform)

    # let's visualize the data
    # Iterate over the Dataloader
    for batch in train_dataloader:
        print(batch)
        imgs = batch['image']
        labels = batch['labels']
        print(imgs.shape)
        print(labels)

        plt.imshow(imgs[0])
        plt.axis('off')
        plt.show()
        break
    

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device) # put model in device (GPU or CPU)
    print(model)

    loss_fn = nn.CrossEntropyLoss() # already includes the Softmax activation
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    num_epochs = 10
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}
    best_val_loss = np.inf

    print("Start training...")
    
    for t in range(num_epochs):
        print(f"\nEpoch {t+1}")
        train_loss, train_acc = epoch_iter(train_dataloader, model, loss_fn, optimizer)
        print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

        val_loss, val_acc = epoch_iter(validation_dataloader, model, loss_fn, is_train=False)
        print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

        # save model when val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, 'best_model.pth')

            # save latest model
            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, 'latest_model.pth')

            # save training history for plotting purposes
            train_history["loss"].append(train_loss)
            train_history["accuracy"].append(train_acc)

            val_history["loss"].append(val_loss)
            val_history["accuracy"].append(val_acc)
        
    print("Finished")

    plotTrainingHistory(train_history, val_history)

    # Evaluate the model in the test set

    # load the best model (i.e. model with the lowest val loss...might not be the last model)
    # we could also load the optimizer and resume training if needed

    model = NeuralNetwork().to(device)
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model'])

    test_loss, test_acc = epoch_iter(test_dataloader, model, loss_fn, is_train=False)
    print(f"\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}")

    showErrors(model, test_dataloader)


