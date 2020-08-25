import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_model_torch(in_shape, out_shape):
    from models import model_torch
    return model_torch(in_shape, out_shape)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def visualize_conf_matrix(matrix, class_list):
    df_cm = pd.DataFrame(matrix, index = [i for i in class_list],
                    columns = [i for i in class_list])
    plt.figure(figsize = (13,7))
    sn.set(font_scale=1.8)
    sn.heatmap(df_cm, annot=True, cmap='Greys', fmt='g', annot_kws={"size": 20})
    plt.show(block=False); 
    folder = './sound_event_detection/figures/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + 'confusion_matrix' + '.png', bbox_inches='tight')

def get_conf_matrix(y_pred, y_test):
    y_pred_max = []
    y_test_max = []
    for j in y_pred:
        y_pred_max.append(np.argmax(j))
    for j in y_test:
        y_test_max.append(np.argmax(j))
    return confusion_matrix(y_test_max, y_pred_max)

def get_metrics(conf_matrix):
    tn = 0.0
    fp = 0.0
    tp = 0.0
    fn = 0.0
    epsilon = 0.01
    for it1 in range(conf_matrix.shape[0]):
        tp += conf_matrix[it1][it1]
        for it2 in range(conf_matrix.shape[1]):
            if it2 != it1:
                fp += conf_matrix[it2][it1]
                fn += conf_matrix[it1][it2]
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision*recall) / (precision + recall + epsilon)
    return tp, precision, recall, f1

print('#'*40, "\n\t\tTesting\n")
token = ''
    
test_reader = pd.read_table('./sound_event_detection/src/test'+token+'.csv', sep='\t', encoding='utf-8')
file_test_df = pd.DataFrame(test_reader)
testfeatures = file_test_df.iloc[:, 1:-1]
testlabel = file_test_df.iloc[:, -1:]
#
X_test = np.array(testfeatures)
y_test = np.array(testlabel)
#
lb = LabelEncoder()
lb_fit = lb.fit_transform(y_test.ravel())
y_test = to_categorical(lb_fit, 9)
#
x_testcnn = np.expand_dims(X_test, axis=2)

device = torch.device('cpu')#'cuda:0' if torch.cuda.is_available() else 'cpu')
print("Executing model on:", device)
model = get_model_torch(x_testcnn.shape[1], 9)

print("Loading model weights...")
PATH = "./sound_event_detection/checkpoint/torch_model.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
prev_loss = checkpoint['loss']
model.eval()
#model.to(device)
print("Loading finished.")

y_pred = model(torch.from_numpy(x_testcnn).float())
y_pred = y_pred.data.cpu().numpy()
print("Shape of prediction:", y_pred.shape)

conf_matrix = get_conf_matrix(y_pred, y_test)
print("Visualizing confusion matrix...")
visualize_conf_matrix(conf_matrix, lb.classes_)
print("Visual of confusion matrix is saved to ./sound_event_detection/figures/confusion_matrix.png")

tp, precision, recall, f1 = get_metrics(conf_matrix)
accuracy = 100 * tp / len(x_testcnn)
print('\nTesting metrics:\n\taccuracy:\t%.3f,\n\tavg. precision:\t%.3f,\n\tavg. recall:\t%.3f,\n\tavg. F1:\t%.3f.\n' % (accuracy,  precision,  recall,  f1))
print('\t\tFinished Testing.')
print('#'*40)