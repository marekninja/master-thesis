from enmt.quant_helper import QuantizationMode, get_quant_backend

from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ModelWrapper():
    """ Stores HF model and tokenizer

    Supports:
    *   reset
    *   creation of quantized model
    """

    def __init__(self, pretrained_model_name_or_path: str, model=None, tokenizer=None, isQuantized=None) -> None:
        """Init of pretrained HF model and tokenizer.

        Args:
            pretrained_model_name_or_path (str): Name of the model from the HF hub
            model ([type], optional): Only for quantization purposes. Defaults to None.
            tokenizer ([type], optional): Only for quantization purposes. Defaults to None.
            isQuantized ([type], optional): Only for quantization purposes. Defaults to None.
        """
        if pretrained_model_name_or_path is not None:
            self.pretrained_model_name_or_path = pretrained_model_name_or_path

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path)
            self.isQuantized = False
            print(
                f"Created model {pretrained_model_name_or_path} succesfully!")
        else:
            self.pretrained_model_name_or_path = None,
            self.model = model
            self.tokenizer = tokenizer
            self.isQuantized = isQuantized

    def quantize(self, mode: QuantizationMode):
        """Quantizes self.model.

        Args:
            mode (QuantizationMode): Mode to use for quantization
        """
        self.model = _makeQuantized(self, mode).model
        self.isQuantized = True
        print(
            f"{self.pretrained_model_name_or_path} quantized using {mode} succesfully!")

    def _get_quantized(self, mode: QuantizationMode):
        """Creates and returns quantized ModelWrapper

        Args:
            mode (QuantizationMode): Mode to use for quantization

        Returns:
            ModelWrapper: Copy of current ModelWrapper with quantized model
        """
        return _makeQuantized(self, mode)

    def reset(self):
        """Resets the model. Model can be trained from scratch.
        """
        config = self.model.config
        model_type = type(self.model)
        self.model = model_type(config)


def _makeQuantized(model_wrapped: ModelWrapper, mode: QuantizationMode) -> ModelWrapper:
    if not isinstance(mode, QuantizationMode):
        raise TypeError(f"mode '{mode}' is not instance of QuantizationMode")

    if mode == QuantizationMode.DYNAMIC:
        quantized_model = _dynamic_quant(model_wrapped)
        return ModelWrapper(None, quantized_model, model_wrapped.tokenizer, True)

    if mode == QuantizationMode.STATIC:
        raise NotImplementedError(
            f"Quantization mode: {mode} not supportted yet.")
    if mode == QuantizationMode.QUANT_AWARE:
        raise NotImplementedError(
            f"Quantization mode: {mode} not supportted yet.")


def _dynamic_quant(model_wrapped: ModelWrapper):
    print(
        f"Using '{get_quant_backend}' engine for quantization")

    quantized_model = torch.quantization.quantize_dynamic(
        model_wrapped.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
