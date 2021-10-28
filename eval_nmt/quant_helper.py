import torch
from enum import Enum


class QuantizationMode(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    QUANT_AWARE = "qa"


def get_quant_backend():
    return torch.backends.quantized.engine
