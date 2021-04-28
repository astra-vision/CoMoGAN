"""
This enables dynamic loading of models, similarly to what happens with the dataset.
"""

import importlib
from networks.base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module "networks/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "networks." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_model_options(model_name):
    model_filename = "networks." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    for name, cls in modellib.__dict__.items():
        if name.lower() == 'modeloptions':
            return cls
    return None

def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from networks import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    return instance
