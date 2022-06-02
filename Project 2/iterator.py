import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Custom imports
from utils import Utils
from config import Config

class Iterator:
    @staticmethod
    def epoch_iterator(dataloader, model, loss_function, optimizer=None, is_train=True):
        if is_train: assert optimizer is not None, 'When training, please provide an optimizer.'
        
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
                X, y = data['image'].to(Config.device), data['labels'].type(torch.LongTensor).to(Config.device)

                # Compute prediction error
                pred = model(X)
                loss = loss_function(pred, y)

                # Backpropagation
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Save training metrics
                total_loss += loss.item()

                probs = F.softmax(pred, dim=1)
                final_pred = torch.argmax(probs, dim=1)

                # TODO: for multilabel
                # threshold = 0.5
                # final_pred = np.array([[1 if i > threshold else 0 for i in j] for j in probs])
                # final_pred = torch.from_numpy(final_pred)
                
                preds.extend(final_pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

        return total_loss / num_batches, accuracy_score(labels, preds)

    @staticmethod
    def train(model, train_dataloader, validation_dataloader, loss_fn):
        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': []}

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        best_val_loss = np.inf
        print("Start training...")

        for t in range(Config.num_epochs):
            print(f"\nEpoch {t+1}")
            train_loss, train_acc = Iterator.epoch_iterator(train_dataloader, model, loss_fn, optimizer)
            print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

            val_loss, val_acc = Iterator.epoch_iterator(validation_dataloader, model, loss_fn, is_train=False)
            print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
                torch.save(save_dict, './pth_models/' + Config.model_name + '_best_model.pth')

            # Save latest model
            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, './pth_models/' + Config.model_name + '_latest_model.pth')

            # Values for plotting
            train_history["loss"].append(train_loss)
            train_history["accuracy"].append(train_acc)
            val_history["loss"].append(val_loss)
            val_history["accuracy"].append(val_acc)
        
        print("Finished")
        return train_history, val_history

    @staticmethod
    def test(model, test_data, loss_function):
        # Load the best model (i.e. model with the lowest val loss...might not be the last model)
        # We could also load the optimizer and resume training if needed
        model = model.to(Config.device)
        checkpoint = torch.load('./pth_models/' + Config.model_name + '_best_model.pth')
        model.load_state_dict(checkpoint['model'])

        test_loss, test_acc = Iterator.epoch_iterator(test_data, model, loss_function, is_train=False)
        print(f"\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}")

        Utils.display_predictions(model, test_data)
