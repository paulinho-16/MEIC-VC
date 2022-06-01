import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from neural_network import ConvolutionalNeuralNetwork

class Train():
    def __init__(self, device, model, loss_fn, optimizer, num_epochs, train_data, validation_data, test_data):
        self.device = device
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        self.train()
        self.test()

        # self.loss_fn = nn.CrossEntropyLoss() # already includes the Softmax activation
        # self.loss_fn = nn.BCELoss()

        # self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        # self.optimizer = optim.Adam(model.parameters(), lr=lr)

    # Analyse training evolution
    def plotTrainingHistory(self, train_history, val_history):
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

    def showErrors(self, model, dataloader, num_examples=20):    
        plt.figure(figsize=(15, 15))

        for ind, data in enumerate(tqdm(dataloader)):
            if (ind % 100 == 0) and (ind != 0): plt.show()
            ind = ind % 100

            inputs, label = data['image'], data['labels']

            # Expected all tensors to be on the same device
            label = label.type(torch.LongTensor)
            inputs, label = inputs.to(self.device), label.to(self.device)
            
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

    def epoch_iter(self, dataloader, model, loss_fn, optimizer=None, is_train=True):
        if is_train:
            assert optimizer is not None, 'When training, please provide an optimizer.'
        
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
                X, y = data['image'].to(self.device), data['labels'].type(torch.LongTensor).to(self.device)

                # Compute prediction error
                pred = model(X)
                # pred = torch.sigmoid(pred) # TODO: multilabel
                # final_pred = torch.argmax(probs, dim=1) # only on lower versions
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

                # TODO: for multilabel
                # threshold = 0.5
                # final_pred = np.array([[1 if i > threshold else 0 for i in j] for j in probs])
                # final_pred = torch.from_numpy(final_pred)
                
                preds.extend(final_pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

        return total_loss / num_batches, accuracy_score(labels, preds)

    def train(self):
        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': []}
        best_val_loss = np.inf
        print("Start training...")
        for t in range(self.num_epochs):
            print(f"\nEpoch {t+1}")
            train_loss, train_acc = self.epoch_iter(self.train_data, self.model, self.loss_fn, self.optimizer)
            print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")
            val_loss, val_acc = self.epoch_iter(self.validation_data, self.model, self.loss_fn, is_train=False)
            print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

            # Save model when val loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': t}
                torch.save(save_dict, 'best-model.pth')

            # Save latest model
            save_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, 'latest-model.pth')

            # Save training history for plotting purposes
            train_history["loss"].append(train_loss)
            train_history["accuracy"].append(train_acc)

            val_history["loss"].append(val_loss)
            val_history["accuracy"].append(val_acc)

        print("Finished")

        self.plotTrainingHistory(train_history, val_history)

    # Evaluate the model in the test set
    def test(self):
        # Load the best model (i.e. model with the lowest val loss...might not be the last model)
        # We could also load the optimizer and resume training if needed
        model = self.model.to(self.device)
        checkpoint = torch.load('best-model.pth')
        model.load_state_dict(checkpoint['model'])

        test_loss, test_acc = self.epoch_iter(self.test_data, model, self.loss_fn, is_train=False)
        print(f"\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}")

        self.showErrors(model, self.test_data)
