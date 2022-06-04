import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config

class Utils:
    @staticmethod
    def calculate_output_size(input_size):
        return (input_size - Config.kernel_size + 2*Config.padding) / Config.stride + 1

    @staticmethod
    def learning_curve_graph(train_history, val_history):
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
    
    @staticmethod
    def display_predictions(model, data, multilabel=False):    
        plt.figure(figsize=(15, 15))

        for ind, data in enumerate(tqdm(data)):
            if (ind % 100 == 0) and (ind != 0): plt.show()
            ind = ind % 100

            ######
            # Save this somewhere else
            ######
            inputs, labels = data['image'], data['labels']
            labels = labels.type(torch.LongTensor) 
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            pred = model(inputs)
            probs = F.softmax(pred, dim=1)
            final_pred = torch.argmax(probs, dim=1)

            if multilabel:
                threshold = 0.3
                final_pred = np.array([[1 if i > threshold else 0 for i in j] for j in probs])
                final_pred = torch.from_numpy(final_pred)

            inputs = inputs[0].cpu()
            
            ######
            # Save this somewhere else
            ######
            if multilabel:
                plt.subplot(10, 10, ind + 1)
                plt.axis("off")
                labels = [idx for idx, label in enumerate(labels[0]) if label.item() == 1]
                preds = [idx for idx, pred in enumerate(final_pred[0]) if pred.item() == 1]
                for i, label in enumerate(labels):
                    plt.text(50*i, -1, label, fontsize=14, color='green') # correct
                for j, pred in enumerate(preds):
                    plt.text(50*(i+1) + 50*j, -1, pred, fontsize=14, color='red') # predicted
            else:
                plt.subplot(10, 10, ind + 1)
                plt.axis("off")
                plt.text(0, -1, labels[0].item(), fontsize=14, color='green') # correct
                plt.text(100, -1, final_pred[0].item(), fontsize=14, color='red') # predicted

            plt.imshow(inputs.permute(1, 2, 0).numpy())
        plt.show()