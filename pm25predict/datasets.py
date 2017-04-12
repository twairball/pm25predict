import numpy as np
import pandas as pd

def split_datasets(dataset, ratio=0.8):
    """
    Split into train and test sets
    """
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return train, test

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


class Dataset():
    def __init__(self, df):
        # keep max for normalizing and inverting
        self.df_scales = df.max()
        self.df = df
    
    def normalized_values(self):
        df_norm = self.df / self.df_scales
        vals = df_norm.values
        # reshape if 1-D series
        if len(vals.shape) == 1:
            vals = np.reshape(vals, (vals.shape[0], 1))
        return vals

class DatasetLoader():
    """
    Dataset loader for model experimentation
    :param df_feats: dataframe of features
    :param df_labels: dataframe of labels, expects single column
    :param look_back: LSTM look_back dimension for reshaping data
    :param ratio: ratio of training data vs test data. Default = 0.9
    """
    def __init__(self, df_feats, df_labels, look_back=3, ratio=0.9):
        self.look_back = look_back
        self.ratio = ratio
        # features dataset
        self.ds_feats = Dataset(df_feats)        
        # labels dataset
        self.ds_labels = Dataset(df_labels)

    def get_datasets(self):
        # get train set
        train, test = split_datasets(self.ds_feats.normalized_values(), self.ratio)
        # get labels
        train_labels, test_labels = split_datasets(self.ds_labels.normalized_values(), self.ratio)

        # create datasets with lookback
        train_feats, train_labels = lookback_dataset(train, train_labels, self.look_back)
        test_feats, test_labels = lookback_dataset(test, test_labels, self.look_back)

        return train_feats, train_labels, test_feats, test_labels
    
    def label_scale(self):
        return self.ds_labels.df_scales[0] or self.ds_labels.df_scales

    def get_labels(self):
        """
        Returns (un-normlized) labels
        """
        train_feats, train_labels, test_feats, test_labels = self.get_datasets()
        # revert scaling
        train_labels = train_labels * self.label_scale()
        test_labels = test_labels * self.label_scale()
        return train_labels, test_labels

