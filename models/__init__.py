from models.predictive_model_v8 import PredictiveModelV8
from models.vq_vae_predictive_model import VQVAEPredictiveModel
from models.autoencoder_model_v0 import AutoEncoderModelV0
from models.autoencoder_model_v1 import AutoEncoderModelV1
from models.base_predictive_model import BasePredictiveModel
from typing import Type

def get_model_class(model_type) -> Type[BasePredictiveModel]:
    model_classes = {
        "PredictiveModelV8": PredictiveModelV8,
        "VQVAEPredictiveModel": VQVAEPredictiveModel,
        "AutoEncoderModelV0": AutoEncoderModelV0,
        "AutoEncoderModelV1": AutoEncoderModelV1
        # Add more models here ...
    }
    return model_classes.get(model_type)