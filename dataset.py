import numpy as np

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

