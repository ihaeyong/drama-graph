import os
import torch
import pickle
import gensim
import numpy as np
import pandas as pd
import os
import sys

from config import model_config as config
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.metrics import multilabel_confusion_matrix, jaccard_score, f1_score, precision_score, recall_score

import itertools
import matplotlib.pyplot as plt


def generate_word_embeddings(vocab):
    if not os.path.exists('{}gensim.glove.6B.{}d.txt'.format(
            config['embeddings_dir'], config['embedding_dim'])):
        glove2word2vec(glove_input_file='{}glove.6B.{}d.txt'.format(
            config['embeddings_dir'], config['embedding_dim']),
            word2vec_output_file='{}gensim.glove.6B.{}d.txt'.format(
            config['embeddings_dir'], config['embedding_dim']))

    embeddings_all = gensim.models.KeyedVectors.load_word2vec_format(
        '{}gensim.glove.6B.{}d.txt'.format(config['embeddings_dir'],
                                           config['embedding_dim']))
    print('Loaded original embeddings')

    # initialize word embeddings matrix
    combined_word_embeddings = np.zeros((vocab.size,
                                         config['embedding_dim']))
    for index, word in vocab.index2word.items():
        try:
            if index < 4:  # deal with special tokens
                combined_word_embeddings[index] = np.random.normal(
                    size=(config['embedding_dim'], ))
                continue
            combined_word_embeddings[index] = embeddings_all[word]
        except KeyError as e:
            # print('KeyError triggered for {}'.format(word))
            combined_word_embeddings[index] = np.random.normal(
                size=(config['embedding_dim'], ))
    print('Created combined + filtered embeddings.')
    with open('{}saved_{}d_word_embeddings.pkl'.format(
            config['embeddings_dir'], config['embedding_dim']), 'wb') as f:
        pickle.dump(combined_word_embeddings, f)
    combined_word_embeddings = torch.from_numpy(combined_word_embeddings).float()
    return combined_word_embeddings


def load_word_embeddings():
    with open('{}saved_{}d_word_embeddings.pkl'.format(
            config['embeddings_dir'], config['embedding_dim']), 'rb') as f:
        combined_word_embeddings = pickle.load(f)
        return torch.from_numpy(combined_word_embeddings).float()


def zero_padding(l, fillvalue=config['<PAD>']):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value=config['<PAD>']):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def input_var(l, vocab):
    indexes_batch = [indexes_from_sentence(vocab, sentence) for sentence in l]
    for idx, indexes in enumerate(indexes_batch):
        indexes_batch[idx] = indexes_batch[idx] + [config['<EOS>']]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def indexes_from_sentence(vocab, sentence):
    indexes = []
    for word in sentence.strip().split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError as e:
            indexes.append(config['<UNK>'])
    return indexes[:config['max_sequence_length']]


def load_data(batched=True, test=False, file_dir=config['file_dir']):
    # Load vocab
    with open(config['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)

    bs = config['batch_size']
    ftype = 'test' if test else 'train'

    df = pd.read_csv('{}text_{}.csv'.format(file_dir, ftype))
    data = (np.array(list(df['transcription'])), np.array(df['label']))

    data = list(zip(data[0], data[1]))
    data.sort(key=lambda x: len(x[0].split()), reverse=True)

    n_iters = len(data) // bs
    batches = []

    for i in range(1, n_iters + 1):
        input_batch = []
        output_batch = []
        for e in data[bs * (i-1):bs * i]:
            input_batch.append(e[0])
            if e[1][0] == '[':
                output_batch.append([int(x) for x in e[1][1:-1].split(',')])
            else:
                output_batch.append([int(x) for x in e[1][2:-2].split(',')])
        inp, lengths = input_var(input_batch, vocab)
        batches.append([inp, lengths, torch.LongTensor(output_batch)])

    return batches


def evaluate(targets, predictions, threshold):
    results = []
    for prediction in predictions:
        result = []
        for score in prediction:
            if score > threshold:
                result.append(1)
            else:
                result.append(0)
        results.append(result)
    performance = {
        'jaccard': jaccard_score(targets, results, average='micro'),
        'f1': f1_score(targets, results, average='micro'),
        'precision': precision_score(targets, results, average='micro'),
        'recall': recall_score(targets, results, average='micro')}
    return performance
