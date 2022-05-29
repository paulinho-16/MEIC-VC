import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

from neural_network import ConvolutionalNeuralNetwork

class Train():
    def __init__(self, device, model, train_data, validation_data, test_data):
        self.device = device
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.loss_fn = nn.CrossEntropyLoss() # already includes the Softmax activation
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        self.train()

    def epoch_iter(self, dataloader, model, loss_fn, optimizer=None, is_train=True):
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

        print('-------------')

        with torch.set_grad_enabled(is_train):
            for batch, (X, y) in enumerate(tqdm(dataloader)):
                print('#################')
                print(f'x: {X}; y: {y}')

                X, y = X.to(self.device), y.to(self.device)

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

    def train(self):
        num_epochs = 10
        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': []}
        best_val_loss = np.inf
        print("Start training...")
        for t in range(num_epochs):
            print(f"\nEpoch {t+1}")
            train_loss, train_acc = self.epoch_iter(self.train_data, self.model, self.loss_fn, self.optimizer)
            print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")
            val_loss, val_acc = self.epoch_iter(self.validation_data, self.model, self.loss_fn, is_train=False)
            print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

            # save model when val loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': t}
                torch.save(save_dict, 'best_model.pth')

            # save latest model
            save_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, 'latest_model.pth')

            # save training history for plotting purposes
            train_history["loss"].append(train_loss)
            train_history["accuracy"].append(train_acc)

            val_history["loss"].append(val_loss)
            val_history["accuracy"].append(val_acc)

        print("Finished")

        # Analyse training evolution
        # def plotTrainingHistory(train_history, val_history):
        #     plt.subplot(2, 1, 1)
        #     plt.title('Cross Entropy Loss')
        #     plt.plot(train_history['loss'], label='train')
        #     plt.plot(val_history['loss'], label='val')
        #     plt.legend(loc='best')

        #     plt.subplot(2, 1, 2)
        #     plt.title('Classification Accuracy')
        #     plt.plot(train_history['accuracy'], label='train')
        #     plt.plot(val_history['accuracy'], label='val')

        #     plt.tight_layout()
        #     plt.legend(loc='best')
        #     plt.show()

        # plotTrainingHistory(train_history, val_history)

        # Evaluate the model in the test set

        # load the best model (i.e. model with the lowest val loss...might not be the last model)
        # we could also load the optimizer and resume training if needed

        model = ConvolutionalNeuralNetwork().to(self.device)
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model'])

        test_loss, test_acc = self.epoch_iter(self.test_data, model, self.loss_fn, is_train=False)
        print(f"\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}")

        # def showErrors(model, dataloader, num_examples=20):    
        #     plt.figure(figsize=(15, 15))

        #     for ind, (X, y) in enumerate(dataloader):
        #         if ind >= num_examples: break
        #         X, y = X.to(self.device), y.to(self.device)
        #         pred = model(X)
        #         probs = F.softmax(pred, dim=1)
        #         final_pred = torch.argmax(probs, dim=1)

        #         plt.subplot(10, 10, ind + 1)
        #         plt.axis("off")
        #         plt.text(0, -1, y[0].item(), fontsize=14, color='green') # correct
        #         plt.text(8, -1, final_pred[0].item(), fontsize=14, color='red')  # predicted
        #         plt.imshow(X[0][0,:,:].cpu(), cmap='gray')
        #         plt.show()

        # showErrors(model, self.test_data)