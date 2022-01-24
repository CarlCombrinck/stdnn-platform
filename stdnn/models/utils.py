from importlib import import_module
from importlib.util import find_spec
from datetime import datetime

class ClassNotFoundError(ImportError):
    """
    Error raised when class not found
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ModelFileNotFoundError(FileNotFoundError):
    """
    Error raised when model file cannot be found
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor for ModelFileNotFoundError
        """
        super().__init__(*args, **kwargs)

def timed(operation_name):
    """Parameterized decorator for timing a model operation (e.g. training/testing)

    Parameters
    ----------
    operation_name : str
        Name of the operation the function represents

    Returns
    -------
    function
        The decorator into which the operation is passed
    """
    def decorator(function):
        def wrapper(self, *args, **kwargs):
            start = datetime.now().timestamp()
            output = function(self, *args, **kwargs)
            end = datetime.now().timestamp()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("{} Time: ""{:0>2}:{:0>2}:{:05.2f}".format(operation_name, int(hours), int(minutes), seconds))
            return output
        return wrapper
    return decorator

def get_model_class(module_name, model_name):
    """
    Utility to return the model class given its name and module

    Parameters
    ----------
    module_name : str
        Name of module in which model class definition is located
    model_name : str
        Name of model class

    Returns
    -------
    class
        The class of the model

    Raises
    ------
    ModuleNotFoundError
        When specified module cannot be found 
    ClassNotFoundError
        When module found but not the specified class
    """
    module_spec = find_spec(f"stdnn.models.{module_name}")
    if module_spec is None:
        raise ModuleNotFoundError(f"No module with name '{module_name}'")
    model_module = import_module(f"stdnn.models.{module_name}")
    model = model_module.__dict__.get(model_name)
    if model is None:
        raise ClassNotFoundError(f"No class with name '{model_name}' in module '{module_name}'")
    return model

