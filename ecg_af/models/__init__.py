from .cnn_lstm import AFCNNLSTM1D
from .ecg_founder import AFECGFounder1D
from .noise_detector import NoiseDetectorModel
from .resnet1d import AFResNet1D


def get_model_registry():
    return {
        "cnn_lstm": AFCNNLSTM1D,
        "resnet1d": AFResNet1D,
        "ecg_founder": AFECGFounder1D,
        "noise_detector": NoiseDetectorModel,
    }
