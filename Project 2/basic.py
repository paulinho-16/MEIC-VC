import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import transforms, models
import matplotlib.pyplot as plt
from dataset import ImageClassificationDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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

def train(model, model_name, num_epochs, train_dataloader, validation_dataloader, loss_fn, optimizer):
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
        torch.save(save_dict, model_name + '_best_model.pth')

      # save latest model
      save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
      torch.save(save_dict, model_name + '_latest_model.pth')

      # save training history for plotting purposes
      train_history["loss"].append(train_loss)
      train_history["accuracy"].append(train_acc)

      val_history["loss"].append(val_loss)
      val_history["accuracy"].append(val_acc)
      
  print("Finished")
  return train_history, val_history

class BasicVersion():
    def __init__(self, model_name, train_dl, validation_dl, test_dl):
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, 4)
            model.to(device)
            # print(model)

            num_epochs = 20

            loss_fn = nn.CrossEntropyLoss() # already includes the Softmax activation
            optimizer_vgg = torch.optim.SGD(model.parameters(), lr=1e-3)

            vgg_train_history, vgg_val_history = train(model, model_name, num_epochs, train_dl, validation_dl, loss_fn, optimizer_vgg)

            plotTrainingHistory(vgg_train_history, vgg_val_history)
            
            self.test(model, test_dl, loss_fn)
        elif model == 'ResNet':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 10)
            model.to(device)

    def test(self, best_model, test_dl, loss_fn):
        model = best_model 
        
        """models.vgg16(pretrained=True).to(device)
        checkpoint = torch.load(best_model)
        model.load_state_dict(checkpoint['model'])"""

        test_loss, test_acc = epoch_iter(test_dataloader, model, loss_fn, is_train=False)
        print(f"\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}")

        showErrors(model, test_dl)
# load model from torchvision (with pretrained=True)

# change the number of neurons in the last layer to the number of classes of the problem at hand (CIFAR10 dataset)
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
    transforms.Resize((200, 200)), 
    transforms.ToTensor()
])

test_transform = transforms.Compose([ # TODO: try another values/transformations
    transforms.ToPILImage(),
    transforms.Resize((200, 200)), # TODO: acho que não devia ter resize (ver link)
    transforms.ToTensor()
])

if __name__ == "__main__":
    # now we need to define a Dataloader, which allows us to automatically batch our inputs, do sampling and multiprocess data loading
    batch_size = 4
    num_workers = 2 # how many processes are used to load the data

    val_train_images = read_images('train.txt')
    test_images = read_images('test.txt')

    train_ratio = int(0.8 * len(val_train_images))
    validation_ratio = len(val_train_images) - train_ratio

    train_images = list(val_train_images[:train_ratio])
    validation_images = list(val_train_images[-validation_ratio:])

    train_data =ImageClassificationDataset(train_images, train_transform)
    validation_data = ImageClassificationDataset(validation_images, validation_transform)
    test_data = ImageClassificationDataset(test_images, test_transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=16, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)

    BasicVersion("vgg16", train_dataloader, validation_dataloader, test_dataloader)