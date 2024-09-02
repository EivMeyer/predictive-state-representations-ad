from models.predictive_model_v8 import PredictiveModelV8
from models.predictive_model_v8_lightweight import PredictiveModelV8Lightweight
from models.base_predictive_model import BasePredictiveModel
from typing import Type

def get_model_class(model_type) -> Type[BasePredictiveModel]:
    model_classes = {
        "PredictiveModelV8": PredictiveModelV8,
        "PredictiveModelV8Lightweight": PredictiveModelV8Lightweight
        # Add more models here ...
    }
    return model_classes.get(model_type)