import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class Utils:
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
    def display_predictions(model, data):    
        plt.figure(figsize=(15, 15))

        for ind, data in enumerate(tqdm(data)):
            if (ind % 100 == 0) and (ind != 0): plt.show()
            ind = ind % 100

            ######
            # Save this somewhere else
            ######
            inputs, label = data['image'], data['labels']
            label = label.type(torch.LongTensor) 
            inputs, label = inputs.to('cuda'), label.to('cuda')
            
            pred = model(inputs)
            probs = F.softmax(pred, dim=1)
            final_pred = torch.argmax(probs, dim=1)

            inputs = inputs[0].cpu()
            ######
            # Save this somewhere else
            ######

            plt.subplot(10, 10, ind + 1)
            plt.axis("off")
            plt.text(0, -1, label[0].item(), fontsize=14, color='green') # correct
            plt.text(100, -1, final_pred[0].item(), fontsize=14, color='red')  # predicted

            plt.imshow(inputs.permute(1, 2, 0).numpy())
        plt.show()