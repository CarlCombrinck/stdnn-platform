import torch.nn as nn
from abc import ABC, abstractmethod

class STModelManager(ABC):
    """
    Abstract spatial-temporal model from which all models must be derived
    """
    def __init__(self):
        self.model = None  

    def set_model(self, model):
        self.model = model

    def has_model(self):
        return self.model is None

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def test_model(self, *args, **kwargs):
        pass

