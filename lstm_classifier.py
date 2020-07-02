import torch
import pickle
import numpy as np
import torch.nn as nn
from torch import optim

import os
import sys

from utils import load_word_embeddings


class LSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.dropout = config['dropout']
        self.n_layers = config['n_layers']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.bidirectional = config['bidirectional']

        self.embedding = nn.Embedding.from_pretrained(
            load_word_embeddings(), freeze=False)

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.n_directions = 2 if self.bidirectional else 1
        self.out = nn.Linear(self.n_directions * self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq, input_lengths):
        max_seq_len, bs = input_seq.size()
        # input_seq =. [max_seq_len, batch_size]
        embedded = self.embedding(input_seq)

        rnn_output, (hidden, _) = self.rnn(embedded)
        rnn_output = torch.cat((rnn_output[-1, :, :self.hidden_dim],
                                rnn_output[0, :, self.hidden_dim:]), dim=1)
        # sum hidden states
        class_scores = self.sigmoid(self.out(rnn_output))

        return class_scores

