import torch
from enum import Enum


class QuantizationMode(Enum):
    """Defines quantization mode

    Args:
        Enum (DYNAMIC): dynamic quantization
        Enum (STATIC): static quantization
        Enum (QUANT_AWARE): quantization aware - model should be fine-tuned/retrained before quantization
    """
    DYNAMIC = "dynamic"
    STATIC = "static"
    QUANT_AWARE = "qa"


def get_quant_backend():
    return torch.backends.quantized.engine
