import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import transforms, models
import matplotlib.pyplot as plt
from dataset import ImageClassificationDataset

from types import SimpleNamespace

config = SimpleNamespace()
config.model_name = "resnet50"
config.num_epochs = 5
config.learning_rate = 0.05
config.batch_size = 16
config.num_workers = 2 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

########################################
######################################## UTILS
########################################

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
    
def read_images(filename):
    images = []
    with open(filename) as file:
        while (line := file.readline().rstrip()):
            images.append(line)
    return images

def showErrors(model, dataloader, num_examples=20):    
    plt.figure(figsize=(15, 15))

    
    for ind, data in enumerate(tqdm(dataloader)):
        if (ind % 100 == 0) and (ind != 0): plt.show()
        ind = ind % 100

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
        plt.text(100, -1, final_pred[0].item(), fontsize=14, color='red')  # predicted
        
        inputs = inputs[0].cpu()
        plt.imshow(inputs.permute(1, 2, 0).numpy())
    plt.show()
    
########################################
######################################## TRAINING
########################################

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
          X, y = data['image'].to(device), data['labels']
          y = y.type(torch.LongTensor) 
          y = y.to(device)
          # Compute prediction error
          pred = model(X)
          loss = loss_fn(pred, y)

          if is_train:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          # Save training metrics
          total_loss += loss.item() # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached

          probs = F.softmax(pred, dim=1)
          final_pred = torch.argmax(probs, dim=1)
          preds.extend(final_pred.cpu().numpy())
          labels.extend(y.cpu().numpy())

    return total_loss / num_batches, accuracy_score(labels, preds)

def train(model, train_dataloader, validation_dataloader, loss_fn):
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    best_val_loss = np.inf
    print("Start training...")

    for t in range(config.num_epochs):
        print(f"\nEpoch {t+1}")
        train_loss, train_acc = epoch_iter(train_dataloader, model, loss_fn, optimizer)
        print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

        val_loss, val_acc = epoch_iter(validation_dataloader, model, loss_fn, is_train=False)
        print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, config.model_name + '_best_model.pth')

        # Save latest model
        save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, config.model_name + '_latest_model.pth')

        # Values for plotting
        train_history["loss"].append(train_loss)
        train_history["accuracy"].append(train_acc)
        val_history["loss"].append(val_loss)
        val_history["accuracy"].append(val_acc)
      
    print("Finished")
    return train_history, val_history

########################################
########################################
########################################

class ClassificationResNet:
    def __init__(self, train_dl, validation_dl):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 4)

        model.to(device)

        loss_function = nn.CrossEntropyLoss() # already includes the Softmax activation

        train_history, val_history = train(model, train_dl, validation_dl, loss_function)
        plotTrainingHistory(train_history, val_history)

        return model, loss_function

class ClassificationVGG16:
    def __init__(self, pre_trained = True):
        self.pre_trained = pre_trained

    def train(self, train_dl, validation_dl):
        model = models.vgg16(pretrained=self.pre_trained)

        model.classifier[6] = nn.Linear(4096, 4)
        model.to(device)

        loss_function = nn.CrossEntropyLoss()

        train_history, val_history = train(model, train_dl, validation_dl, loss_function)
        plotTrainingHistory(train_history, val_history)

        return model, loss_function

    def test(self, best_model, test_dl, loss_fn):
        model = best_model 
        
        """models.vgg16(pretrained=True).to(device)
        checkpoint = torch.load(best_model)
        model.load_state_dict(checkpoint['model'])"""

        test_loss, test_acc = epoch_iter(test_dataloader, model, loss_fn, is_train=False)
        print(f"\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}")

        showErrors(model, test_dl)

########################################
########################################
########################################

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

if __name__ == "__main__":
    # now we need to define a Dataloader, which allows us to automatically batch our inputs, do sampling and multiprocess data loading
    val_train_images = read_images('train.txt')
    test_images = read_images('test.txt')

    train_ratio = int(0.8 * len(val_train_images))
    validation_ratio = len(val_train_images) - train_ratio

    train_images = list(val_train_images[:train_ratio])
    validation_images = list(val_train_images[-validation_ratio:])

    train_data =ImageClassificationDataset(train_images, transforms_dict['train'])
    validation_data = ImageClassificationDataset(validation_images, transforms_dict['validation'])
    test_data = ImageClassificationDataset(test_images, transforms_dict['test'])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=config.batch_size, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)

    neural_network = ClassificationResNet(train_dataloader, validation_dataloader)
    #model, loss_fn = neural_network.train(train_dataloader, validation_dataloader)
    #neural_network.test(model, test_dataloader, loss_fn)