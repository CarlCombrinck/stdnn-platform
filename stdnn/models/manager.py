from abc import ABC, abstractmethod
import os 
import torch

class ModelFileNotFoundError(FileNotFoundError):
    """
    Error raised when model file cannot be found
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class STModelManager(ABC):
    """
    Abstract spatial-temporal model from which all models must be derived
    """
    def __init__(self, model=None):
        self.model = model  

    def set_model(self, model):
        self.model = model

    def has_model(self):
        return self.model is not None

    def save_model(self, model_dir, epoch=None):
        if model_dir is None:
            return
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        epoch = str(epoch) if epoch else type(self.model).__name__
        file_name = os.path.join(model_dir, epoch + '.pt')
        with open(file_name, 'wb') as f:
            torch.save(self.model, f)

    def load_model(self, model_dir, epoch=None):
        if not model_dir:
            return
        epoch = str(epoch) if epoch else type(self.model).__name__
        file_name = os.path.join(model_dir, epoch + '.pt')
        if not os.path.exists(model_dir):
            raise ModelFileNotFoundError(f"Could not locate directory '{model_dir}'")
        if not os.path.exists(file_name):
            raise ModelFileNotFoundError(f"Could not locate model file '{file_name}'")
        with open(file_name, 'rb') as f:
            self.model = torch.load(f)

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def test_model(self, *args, **kwargs):
        pass

