import importlib
import os
from typing import Type
from models.base_predictive_model import BasePredictiveModel

def get_model_class(model_type: str) -> Type[BasePredictiveModel]:
    model_classes = {}
    models_dir = os.path.dirname(__file__)

    for filename in os.listdir(models_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            module = importlib.import_module(f'models.{module_name}')
            
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if (isinstance(attribute, type) and 
                    issubclass(attribute, BasePredictiveModel) and 
                    attribute is not BasePredictiveModel):
                    model_classes[attribute_name] = attribute

    return model_classes.get(model_type)