from .datasets import Dataset, DatasetLoader
from .models import ModelContext

import os
from glob import glob
from copy import copy, deepcopy
from datetime import timedelta, datetime

import h5py
from influxdb import DataFrameClient

class ModelLoader():

    def __init__(self, num_features=13, look_back=3, dirpath='./'):
        self.dirpath = dirpath

        # TODO: how to figure out model dimension from loading from file?
        self.model_context = ModelContext(num_features=num_features, look_back=look_back)
        self.load_latest_model(dirpath)

    def load_latest_model(self, dirpath):
        # look in directory and find newest model 
        files = glob(dirpath + '*.h5')
        if len(files) > 0:
            newest = max(files, key=os.path.getctime)
            self.load_model(newest)
            print("[ModelLoader] loading model: %s" % newest)
        else:
            print("[ModelLoader] no model found at dir: %s" % dirpath)

    def load_model(self, path):
        self.model_context.model.load_weights(path)
    
    def save_model(self, path=None):
        if path == None:
            filename = "%s_model.h5" % datetime.now().strftime("%Y%m%d%H%M")
            path = self.dirpath + filename
        print("[ModelLoader] saving model to %s" % path)
        self.model_context.model.save_weights(path)

class InfluxDataLoader():

    def __init__(self, database='gams', tables=['indoor', 'outdoor']):
        self.db = DataFrameClient(database=database)
        self.tables = tables
        self.df = self.get_dataframe(self.tables)
    
    def get_dataframe(self, tables):
        _tables = deepcopy(tables)

        # get 1st df table
        df = self.df_table(_tables.pop(0))

        # join tables
        for table in _tables: 
            _df = self.df_table(table)
            suffix = "_%s" % table
            df = df.join(_df, rsuffix=suffix)
        return df


    def df_table(self, table):
        query = "select * from %s" % table
        df = self.db.query(query)[table]
        df = hourly(df)
        return df


def hourly(df):
    """
    Sample hourly mean values. 
    For missing intervals we fill forward
    Additionally fill backwards (in case 1st row is missing)
    """
    return df.resample('H').mean().fillna(method='ffill').fillna(method='bfill')