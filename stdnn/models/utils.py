from importlib import import_module
from importlib.util import find_spec

class ClassNotFoundError(ImportError):
    """
    Error raised when class not found
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

