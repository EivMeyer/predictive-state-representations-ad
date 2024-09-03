from models.predictive_model_v8 import PredictiveModelV8
from models.vq_vae_predictive_model import VQVAEPredictiveModel
from models.single_step_predictive_model import SingleStepPredictiveModel
from models.base_predictive_model import BasePredictiveModel
from typing import Type

def get_model_class(model_type) -> Type[BasePredictiveModel]:
    model_classes = {
        "PredictiveModelV8": PredictiveModelV8,
        "VQVAEPredictiveModel": VQVAEPredictiveModel,
        "SingleStepPredictiveModel": SingleStepPredictiveModel,
        # Add more models here ...
    }
    return model_classes.get(model_type)