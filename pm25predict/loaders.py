from .datasets import Dataset, DatasetLoader
from .models import BaseModel

import os
from glob import glob
from copy import copy, deepcopy

from influxdb import DataFrameClient

class ModelLoader():

    def __init__(self, dirpath='.'):
        # TODO: how to figure out model dimension from loading from file?
        self.base_model = BaseModel()
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
        self.base_model.model.load_weights(path)
    
    def save_model(self, path):
        self.base_model.model.save_weights(path)

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


## sample hourly mean
## for missing intervals we fill forward
def hourly(df):
    return df.resample('H').mean().fillna(method='ffill')