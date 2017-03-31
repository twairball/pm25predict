from datasets import Dataset, DatasetLoader
from models import BaseModel


class ModelLoader():

    def __init__(self, dirpath='.'):
        self.base_model = self.load_latest_model(dirpath)
    
    def load_latest_model(self, dirpath):
        # TODO: look in directory and find newest model 
        return BaseModel()

    def load_model(self, path):
        self.base_model.model.load_weights(path)
    
    def save_model(self, path):
        self.base_model.model.save_weights(path)

