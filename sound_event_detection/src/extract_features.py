import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa 
from sklearn.utils import shuffle
from utils import extract_features, read_audio_file
delta = False

def addSmpls(train, test, samples, class_name):
    randf = np.random.rand(len(samples)) > 0.2
    while np.sum(randf) < 0.78*len(samples) or np.sum(randf) > 0.82*len(samples):
        randf = np.random.rand(len(samples)) > 0.2
    print(class_name, ' (train/test):', np.sum(randf), (len(samples) - np.sum(randf)))
    for i in range(0, len(samples)):
        if randf[i] == True:
            train.append((samples[i], class_name))
        else:
            test.append((samples[i], class_name))
    return train, test

def genSets(samples, classes):
    print("Generating sets...\n")
    train = []
    test = []
    class_num = len(classes)
    class_distribution = np.zeros(class_num)
    class_samples = [[] for x in range(class_num)]
    
    for y in samples:
        for cl_idx, cl in enumerate(classes):
            if cl in y:
                class_samples[cl_idx].append(y)
                class_distribution[cl_idx] += 1

    print("Class distribution:", class_distribution)
    print("Partitioning to test and train sets...")
    for idx, cl_list in enumerate(tqdm(class_samples)):
        train, test = addSmpls(train, test, cl_list, classes[idx])

    print("Partitioned:", len(train), len(test))
    return train, test, class_distribution

def get_classes(samples):
    classes = set()
    for sample in samples:
        pos1 = sample.find('"')+1
        pos2 = pos1 + sample[pos1:].find('"')
        class_name = sample[pos1:pos2]
        if class_name not in classes:
            classes.add(class_name)
    return list(classes)


def extr(df, X, sr, bookmark):
    features = extract_features(X, sr, delta=delta)
    df.loc[bookmark] = [features]

samples = os.listdir('./pre_proc/')
print("There are {} samples.".format(len(samples)))

class_list = get_classes(samples)
print('There are {} classes:\n{}', len(class_list), class_list)

train, test, class_distribution = genSets(samples, class_list)

data_len = set()
print("Feature extraction for test set...")
test_df = pd.DataFrame(columns=['feature'])
test_labels = []
for index, sample in tqdm(enumerate(test)):    
    file_name = sample[0]
    file_class = sample[1]
    data, sr = read_audio_file('./pre_proc/'+file_name)
    data_len.add(len(data)/sr)    
    extr(test_df, data, sr, index)
    test_labels.append(sample[1])

test_labels = pd.DataFrame(test_labels)
print("Test Labels length:", len(test_labels))

test_df3 = pd.DataFrame(test_df['feature'].values.tolist())
test_newdf = pd.concat([test_df3, test_labels], axis=1)
print("Test DF length:", len(test_newdf))

test_rnewdf = test_newdf.rename(index=str, columns={"0": "label"})
test_rnewdf = shuffle(test_rnewdf)
test_rnewdf = test_rnewdf.fillna(0)
print("New Test DF length:", len(test_rnewdf))

token = ''
if delta:
    token = '_delta'
print("Saving {} ...".format('test'+token+'.csv'))
test_rnewdf.to_csv('test'+token+'.csv', sep='\t', encoding='utf-8')

data_len = set()
print("Feature extraction for train set...")
train_df = pd.DataFrame(columns=['feature'])
train_labels = []
for index, sample in tqdm(enumerate(train)):    
    file_name = sample[0]
    file_class = sample[1]
    data, sr = read_audio_file('./pre_proc/'+file_name)
    data_len.add(len(data)/sr)    
    extr(train_df, data, sr, index)
    train_labels.append(sample[1])

train_labels = pd.DataFrame(train_labels)
print("Train Labels length:", len(train_labels))

train_df3 = pd.DataFrame(train_df['feature'].values.tolist())
train_newdf = pd.concat([train_df3, train_labels], axis=1)
print("Train DF length:", len(train_newdf))

train_rnewdf = train_newdf.rename(index=str, columns={"0": "label"})
train_rnewdf = shuffle(train_rnewdf)
train_rnewdf = train_rnewdf.fillna(0)
print("New Train DF length:", len(train_rnewdf))

print("Saving {} ...".format('train'+token+'.csv'))
train_rnewdf.to_csv('train'+token+'.csv', sep='\t', encoding='utf-8')



