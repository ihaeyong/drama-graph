def model_keras(in_shape, out_shape):
    import keras
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, Activation, Input, Flatten, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras import regularizers
    #CNN 59.012% adam, 61.235% rmsprop
    model = Sequential()
    model.add(Conv1D(256, 5,padding='same',
                    input_shape=(in_shape,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5,padding='same'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.1))

    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))

    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(out_shape))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    #print(model.summary())
    return model

def model_torch(in_shape, out_shape):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.process_conv = nn.Sequential(nn.Conv1d(in_shape, 256, 3, padding=1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 128, 3, padding=1),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=8),
                                        nn.Conv1d(128, 128, 3, padding=1),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, 3, padding=1),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU())
            ''',
            nn.Conv1d(128, 128, 5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
            '''
            self.process_lin = nn.Sequential(nn.Linear(128, 128),
                                        nn.Linear(128, out_shape),
                                        nn.BatchNorm1d(out_shape),
                                        nn.Softmax(dim=-1))

        def forward(self, x):
            x = self.process_conv(x)
            x = x.view(-1, self.num_flat_features(x))
            x = self.process_lin(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


    net = Net()
    print(net)
    return net