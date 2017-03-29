import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

from keras.layers import TimeDistributed, Activation
from numpy.random import choice

import numpy as np

class Dataset():
    def __init__(self, df, ratio=0.9):
        # keep max for normalizing and inverting
        self.df_scales = df.max()
        self.df = df
        self.ratio = ratio
    
    def normalized_values(self):
        df_norm = self.df / self.df_scales
        return df_norm.values

    def split_datasets(self):
        dataset = self.normalized_values()
        return split_datasets(dataset, self.ratio)

class DatasetPreprocessor():
    def __init__(self, df_feats, df_labels, look_back=3):
        self.look_back = look_back
        # features dataset
        self.ds_feats = Dataset(df_feats)        
        # labels dataset
        self.ds_labels = Dataset(df_labels)

    def get_datasets(self):
        ### TODO: refactor
        # get train set
        train, test = self.ds_feats.split_datasets()
        # get labels
        train_labels, test_labels = self.ds_labels.split_datasets()

        # create datasets with lookback
        train_feats, train_labels = lookback_dataset(train, train_labels, self.look_back)
        test_feats, test_labels = lookback_dataset(test, test_labels, self.look_back)

        return train_feats, train_labels, test_feats, test_labels
    
    def label_scale(self):
        return self.ds_labels.df_scales[0]

    def get_labels(self):
        """
        Returns (un-normlized) labels
        """
        train_feats, train_labels, test_feats, test_labels = self.get_datasets()
        # revert scaling
        train_labels = train_labels * self.label_scale()
        test_labels = test_labels * self.label_scale()
        return train_labels, test_labels



class BaseModel():
    def __init__(self, df_feats, df_labels, look_back=3):
        self.look_back = look_back
        self.num_features = df_feats.shape[1]

        # features dataset
        self.ds_feats = Dataset(df_feats)        
        # labels dataset
        self.ds_labels = Dataset(df_labels)

        # create model
        self.model = create_lstm_model(input_shape=(self.look_back, self.num_features))
    
    def get_datasets(self):
        ### TODO: refactor
        # get train set
        train, test = self.ds_feats.split_datasets()
        # get labels
        train_labels, test_labels = self.ds_labels.split_datasets()

        # create datasets with lookback
        train_feats, train_labels = lookback_dataset(train, train_labels, self.look_back)
        test_feats, test_labels = lookback_dataset(test, test_labels, self.look_back)

        return train_feats, train_labels, test_feats, test_labels

    def label_scale(self):
        return self.ds_labels.df_scales[0]

    def get_labels(self):
        """
        Returns (un-normlized) labels
        """
        train_feats, train_labels, test_feats, test_labels = self.get_datasets()
        # revert scaling
        train_labels = train_labels * self.label_scale()
        test_labels = test_labels * self.label_scale()
        return train_labels, test_labels

    def train(self, nb_epoch=50):
        
        train_feats, train_labels, test_feats, test_labels = self.get_datasets()
        
        # reshape input to be [samples, time steps, features]
        train_feats = reshape_for_lstm(train_feats, self.look_back)

        # train model
        self.model.fit(train_feats, train_labels, nb_epoch=nb_epoch, batch_size=1, verbose=2)
    
    def test(self):

        train_feats, train_labels, test_feats, test_labels = self.get_datasets()
        
        # reshape input to be [samples, time steps, features]
        train_feats = reshape_for_lstm(train_feats, self.look_back)
        test_feats = reshape_for_lstm(test_feats, self.look_back)

        # evaluate model 
        testPredict, trainPredict = evaluate_model(self.model, train_feats, train_labels, test_feats, test_labels)

        # revert scaling
        testPredict = testPredict * self.label_scale()
        return testPredict

##  
## Data preparation
##

def split_datasets(dataset, ratio=0.8):
    """
    Split into train and test sets
    """
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train, test

def create_dataset(dataset, look_back=3):
    """
    Create dataset with look_back
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, ])
    return np.array(dataX), np.array(dataY)

def lookback_dataset(features, labels, look_back=3):
    """
    Create dataset with look_back, in sync with labels
    """
    dataX, dataY = [], []
    for i in range(len(features) - look_back - 1):
        x = features[i:(i+look_back), :]
        dataX.append(x)
        dataY.append(labels[i+look_back,])
    return np.array(dataX), np.array(dataY)

def reshape_for_lstm(x, look_back=3):
    # reshape input to be [samples, time steps, features]
    return np.reshape(x, (x.shape[0], look_back, x.shape[-1]))



##
## Models
##

def create_lstm_model(input_shape=(3,6)):
    model = Sequential()
    model.add(LSTM(8, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model 


from sklearn.metrics import mean_squared_error
import math

def evaluate_model(model, train_feats, train_labels, test_feats, test_labels):
    # make predictions
    trainPredict = model.predict(train_feats, batch_size=1)
    # model.reset_states()
    testPredict = model.predict(test_feats, batch_size=1)
    
    def revert_norm(arr):
        return arr[:,0] if arr.size > len(arr) else arr
    
    # invert predictions back from scaler
    trainPredict = revert_norm(trainPredict)
    testPredict = revert_norm(testPredict)
        
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(train_labels, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test_labels, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    
    # shift by 1 error:
    trainScore = math.sqrt(mean_squared_error(train_labels[:-1], trainPredict[1:]))
    print('(shift by 1) Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test_labels[:-1], testPredict[1:]))
    print('(shift by 1) Test Score: %.2f RMSE' % (testScore))
    
    return testPredict, trainPredict

