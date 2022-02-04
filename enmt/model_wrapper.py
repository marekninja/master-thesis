from enmt.quant_helper import QuantizationMode, get_quant_backend

from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os


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

    def quantizeDynamic(self):
        """Quantizes self.model.

        Args:
            mode (QuantizationMode): Mode to use for quantization
        """
        self.model = _dynamic_quant(self.model)
        self.isQuantized = True
        print(
            f"Dynamic quantization of '{self.pretrained_model_name_or_path}' successful!")

    def quantizeStaticStart(self):
        """Quantizes self.model.

        Args:
            mode (QuantizationMode): Mode to use for quantization
        """
        _static_quant_start(self)
        self.model = _dynamic_quant(self.model)
        self.isQuantized = True
        print(
            f"Started STATIC quantization of '{self.pretrained_model_name_or_path}'..."
            f"Calibration needed...")


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

    def getSize(self) -> float:
        torch.save(self.model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")/1e6
        os.remove('temp.p')
        return size


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
        f"Using '{get_quant_backend()}' engine for quantization")
    model_wrapped.model.to('cpu')
    quantized_model = torch.quantization.quantize_dynamic(
        model_wrapped.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def _static_quant_start(model_wrapped: ModelWrapper):
    print(
        f"Using '{get_quant_backend()}' engine for quantization")
    model_wrapped.model.to('cpu')
    model_wrapped.model.eval()

    # Fuse Conv, bn and relu
    model_wrapped.model.fuse_model()

    num_calibration_batches = 32

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model_wrapped.model.qconfig = torch.quantization.default_qconfig
    print(model_wrapped.model.qconfig)
    torch.quantization.prepare(model_wrapped.model, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', model_wrapped.model.features[1].conv)

    # Calibrate with the training set
    evaluate(model_wrapped.model, criterion, data_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(model_wrapped.model, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
          model_wrapped.model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(model_wrapped.model)

    top1, top5 = evaluate(model_wrapped.model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
