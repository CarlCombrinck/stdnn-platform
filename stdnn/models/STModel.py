import torch.nn as nn
from abc import ABC, abstractmethod

class STModel(nn.Module, ABC):
    """
    Abstract spatial-temporal model from which all models must be derived
    """

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *input, **kwargs):
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def test_model(self, *args, **kwargs):
        pass
