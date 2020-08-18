import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

from sklearn.metrics import confusion_matrix
import os
from tqdm import tqdm

#os.environ['CUDA_VISIBLE_DEVICES'] = "7"
delta = False

def get_model_keras(in_shape, out_shape):
    from models import model_keras

    return model_keras(in_shape, out_shape)

def get_model_torch(in_shape, out_shape):
    from models import model_torch
    return model_torch(in_shape, out_shape)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

token = ''
if delta:
    token = '_delta'
    
train_reader = pd.read_table('train'+token+'.csv', sep='\t', encoding='utf-8')
file_train_df = pd.DataFrame(train_reader)
print('Train'+token+'.csv:\n', file_train_df.head())

trainfeatures = file_train_df.iloc[:, 1:-1]
trainlabel = file_train_df.iloc[:, -1:]

test_reader = pd.read_table('test'+token+'.csv', sep='\t', encoding='utf-8')
file_test_df = pd.DataFrame(test_reader)
print('Test'+token+'.csv:\n', file_test_df.head())

testfeatures = file_test_df.iloc[:, 1:-1]
testlabel = file_test_df.iloc[:, -1:]

#
X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)
#print("Feature train & test shape:", X_train.shape, X_test.shape)
#print("Label train & test shape:", y_train.shape, y_test.shape)

#
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train.ravel()), 9)
y_test = to_categorical(lb.fit_transform(y_test.ravel()), 9)
print("Label train & test shape:", y_train.shape, y_test.shape)

print("save classes...")
np.save('classes.npy', lb.classes_)
#

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)
print("Feature CNN train & test shape:", x_traincnn.shape, x_testcnn.shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training on:", device)
model = get_model_torch(x_traincnn.shape[1], 9)
model.to(device)
best_train_acc = 0
best_test_acc = 0
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
#print("Label train & test first 5 el:\n", y_train[:5], '&', y_test[:5])
train_data = TensorDataset(torch.from_numpy(x_traincnn).float(), torch.from_numpy(y_train).long())
test_data = TensorDataset(torch.from_numpy(x_testcnn).float(), torch.from_numpy(y_test).long())

trainloader = DataLoader(train_data, batch_size=16,
                                        shuffle=True, num_workers=2)
testloader = DataLoader(test_data, batch_size=16,
                                        shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.00001)
epochs = 300
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    counter = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        gt = labels.data
        correct += (predicted == gt).float().sum()
        counter += 1
    accuracy = 100 * correct / len(trainloader.dataset)
    best_train_acc = max(best_train_acc, accuracy)
    print('Training [%d] loss %.3f and accuracy %.3f and prev. best acc. %.3f' % (epoch + 1, running_loss / counter, accuracy, best_train_acc))
    model.eval()
    running_loss = 0.0
    counter = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            gt = labels.data
            correct += (predicted == gt).float().sum()
            counter += 1
        accuracy = 100 * correct / len(testloader.dataset)
        print(' Testing [%d] loss %.3f and accuracy %.3f and prev. best acc. %.3f\n' % (epoch + 1, running_loss / counter, accuracy, best_test_acc))
        if best_test_acc < accuracy:
            best_test_acc = accuracy
            EPOCH = epoch + 1
            folder = '../checkpoint/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            PATH = folder+"torch_model.pt"
            LOSS = running_loss / counter
            torch.save({
                        'epoch': EPOCH,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': LOSS,
                        }, PATH)
print('Finished Training')

