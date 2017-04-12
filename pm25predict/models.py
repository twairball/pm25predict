import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam

from sklearn.metrics import mean_squared_error
import math
import numpy as np
from .datasets import Dataset, DatasetLoader

class ModelContext():
    def __init__(self, num_features=13, look_back=3):
        # LSTM dimensions
        self.look_back = look_back
        self.num_features = num_features

        # create model
        self.model = self.create_model(input_shape=(self.look_back, self.num_features))
    
    def create_model(self, input_shape):
        return create_lstm_model(input_shape)

    def train(self, dataset_loader, batch_size=1, nb_epoch=50):
        """
        Train model from dataset
        """
        train_feats, train_labels, test_feats, test_labels = dataset_loader.get_datasets()
        # reshape input to be [samples, time steps, features]
        train_feats = reshape_for_lstm(train_feats, self.look_back)

        # train model
        self.model.fit(train_feats, train_labels, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)
    
    def update(self, dataset_loader):
        """
        Update model with observations
        """
        self.train(dataset_loader, nb_epoch=1)

    def test(self, dataset_loader):
        """
        Test model vs testing data
        """
        train_feats, train_labels, test_feats, test_labels = dataset_loader.get_datasets()

        # reshape input to be [samples, time steps, features]
        train_feats = reshape_for_lstm(train_feats, self.look_back)
        test_feats = reshape_for_lstm(test_feats, self.look_back)

        # evaluate model 
        testPredict, trainPredict = evaluate_model(self.model, train_feats, train_labels, test_feats, test_labels)
       
        # revert scaling
        testPredict = testPredict * dataset_loader.label_scale()
        return testPredict

    def predict(self, dataset_loader):
        """
        Make a prediction
        """
        train_feats, train_labels, test_feats, test_labels = dataset_loader.get_datasets()

        # reshape input to be [samples, time steps, features]
        test_feats = reshape_for_lstm(test_feats, self.look_back)

        # predict
        testPredict = self.model.predict(test_feats, batch_size=1)

        # take 1st prediction only
        testPredict = testPredict[0]
        
        # revert scaling
        testPredict = testPredict * dataset_loader.label_scale()
        return testPredict


class StackedModel(ModelContext):
    def create_model(self, input_shape):
        return create_stacked_lstm_model(input_shape)


##  
## Data preparation
##

def reshape_for_lstm(x, look_back=3):
    """ reshape input to be [samples, time steps, features] """
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

def create_stacked_lstm_model(input_shape=(3,6)):
    model = Sequential()
    model.add(LSTM(8, input_shape=input_shape, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(16, return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(LSTM(8))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model 


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

