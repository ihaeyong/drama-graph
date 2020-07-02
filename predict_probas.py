import torch
import pickle
import numpy as np
import pandas as pd
import os
import sys

from create_vocab import Vocabulary
from lstm_classifier import LSTMClassifier
from config import model_config as config
from utils import load_data, evaluate



# Load test data

print("load text_test.csv")
test_batches = load_data(test=True)

device = 'cuda:{}'.format(config['gpu']) if \
    torch.cuda.is_available() else 'cpu'

# Load pretrained model
model = LSTMClassifier(config)
model = model.to(device)
checkpoint = torch.load('./major/runs/{}-best_model.pth'.format(config['model_code']),
                        map_location=device)
model.load_state_dict(checkpoint['model'])

with torch.no_grad():
    # Predict
    targets_all = np.empty([0, config['output_dim']])
    predictions_all = np.empty([0, config['output_dim']])
    i = 0
    for batch in test_batches:
        inputs, lengths, targets = batch

        inputs = inputs.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device, dtype=torch.float)

        predictions = model(inputs, lengths)
        predictions = predictions.to(device)

        # evaluate on cpu
        targets = np.array(targets.cpu())
        predictions = np.array(predictions.cpu())
        targets_all = np.append(targets_all, targets, axis=0)
        predictions_all = np.append(predictions_all, predictions, axis=0)
        i += 1
        if i%20 == 0 or i == len(test_batches):
            print("predict test batch {}/{}.".format(i, len(test_batches)))

    with open('./text_lstm_classifier_major.pkl', 'wb') as f:
        pickle.dump(predictions_all, f)

    print("saved prediction result to text_lstm_classifier_major.pkl")

# plot_confusion_matrix(targets_all, predictions_all,
#                       classes=['neutrality', 'happiness', 'sadness', 'fear', 'disgust', 'anger', 'surprise'])

performance1 = evaluate(targets_all, predictions_all, 0.2)
performance2 = evaluate(targets_all, predictions_all, 0.3)
performance3 = evaluate(targets_all, predictions_all, 0.5)

# print(performance1)
print(performance2)
# print(performance3)