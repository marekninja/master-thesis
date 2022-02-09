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
            self.isPrepared = False
            print(
                f"Created model {pretrained_model_name_or_path} succesfully!")
            print(f"Size of model: {self.getSize()}")
        else:
            self.pretrained_model_name_or_path = None,
            self.model = model
            self.tokenizer = tokenizer
            self.isQuantized = isQuantized



    def quantizeDynamic(self, test_tr = True):
        """
        Quantizes self.model.

        Args:
            test_tr: whether to test model translation before and after quantization

        Returns:
            in-place

        """
        if test_tr:
            _test_translation(self)

        self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
            )
        self.isQuantized = True

        if test_tr:
            _test_translation(self)
        print("Size of model after quantization", self.getSize())
        print(
            f"Dynamic quantization of '{self.pretrained_model_name_or_path}' successful!")

    def quantizeStaticStart(self, q_backend='fbgemm', test_tr = True):
        """
        Prepares model for Static quantization
        Args:
            q_backend: backend for quantization
            test_tr: whether to test model translation before quantization

        Returns:
            in-place
        """
        if test_tr:
            _test_translation(self)

        # _static_quant_start(self,q_backend) #inplace
        print(
            f"Using '{q_backend}' engine for quantization")

        # self.model.eval()
        # self.model.to('cpu')

        # Fuse Conv, bn and relu
        # model_wrapped.model.fuse_model()

        # Specify quantization configuration
        # Start with simple min/max range estimation and per-tensor quantization of weights
        self.model.qconfig = torch.quantization.get_default_qconfig(q_backend)
        print(self.model.qconfig)
        torch.quantization.prepare(self.model, inplace=True)

        self.isPrepared = True
        print(
            f"Prepared for STATIC quantization of '{self.pretrained_model_name_or_path}'..."
            f"Calibration needed...")

    def quantizeStaticConvert(self, test_tr= True):
        """
        Finishes Static quantization
        Should be run after Calibration

        Args:
            test_tr: whether to test model translation after quantization

        Returns:
            in-place
        """
        # _static_quant_convert(self)
        # Convert to quantized model
        # self.model.to('cpu')
        # self.model.eval()
        torch.quantization.convert(self.model, inplace=True)

        self.isQuantized = True
        if test_tr:
            _test_translation(self)
        print('Post Training Quantization: Convert done')
        print("Size of model after quantization", self.getSize())

    def quantizeQATStart(self, q_backend='fbgemm', test_tr= True):
        """
        Preparation of model for QAT

        Args:
            q_backend: backend for quantization
            test_tr: whether to test model translation before quantization
        Returns:
            in-place
        """
        if test_tr:
            _test_translation(self)
        self.model.to('cuda')
        # Specify quantization configuration
        # Start with simple min/max range estimation and per-tensor quantization of weights
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(q_backend)
        print(self.model.qconfig)

        torch.quantization.prepare_qat(self.model, inplace=True)
        self.isPrepared = True
        # _qat_prepare(self,q_backend)
        print("Model prepared for QAT. Proceed with training/fine-tuning...")

    def quantizeQATConvert(self, test_tr= True):
        """
        Converts QAT model to INT8
        Args:
            test_tr: whether to test model translation after quantization

        Returns:
            in-place
        """
        # _qat_convert(self)
        self.model.to('cpu')
        # Convert to quantized model
        torch.quantization.convert(self.model, inplace=True)
        self.model.eval()

        print("Size of model after quantization", self.getSize())

        if test_tr:
            _test_translation(self)

        print('Convert done. Can run eval...')



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

        # def reinit_model_weights(m: torch.nn.Module):
        #     if hasattr(m, "children"):
        #         for m_child in m.children():
        #             if hasattr(m_child, "reset_parameters"):
        #                 m_child.reset_parameters()
        #             reinit_model_weights(m_child)

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


def _test_translation(model_wrapped: ModelWrapper):
    # model_wrapped.model.to("cpu")
    # model_wrapped.model.eval()
    tok = model_wrapped.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt",
                          padding=True)
    translated = model_wrapped.model.generate(**tok)
    print("Example translation:",[model_wrapped.tokenizer.decode(t, skip_special_tokens=True) for t in translated])

def _dynamic_quant(model_wrapped: ModelWrapper):
    print(
        f"Using '{get_quant_backend()}' engine for quantization")
    model_wrapped.model.to('cpu')
    quantized_model = torch.quantization.quantize_dynamic(
                model_wrapped.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
            )
    return quantized_model

# def _static_quant_start(model_wrapped: ModelWrapper,q_backend: str):
#     print(
#         f"Using '{q_backend}' engine for quantization")
#     model_wrapped.model.to('cpu')
#     model_wrapped.model.eval()
#
#     # Fuse Conv, bn and relu
#     # model_wrapped.model.fuse_model()
#
#     # Specify quantization configuration
#     # Start with simple min/max range estimation and per-tensor quantization of weights
#     model_wrapped.model.qconfig = torch.quantization.get_default_qconfig(q_backend)
#     print(model_wrapped.model.qconfig)
#     torch.quantization.prepare(model_wrapped.model, inplace=True)
#
# def _static_quant_convert(model_wrapped: ModelWrapper):
#     # Convert to quantized model
#     model_wrapped.model.to('cpu')
#     torch.quantization.convert(model_wrapped.model, inplace=True)
#
# def _qat_prepare(model_wrapped: ModelWrapper,q_backend: str):
#
#     model_wrapped.model.to('cuda')
#     # Specify quantization configuration
#     # Start with simple min/max range estimation and per-tensor quantization of weights
#     model_wrapped.model.qconfig = torch.quantization.get_default_qat_qconfig(q_backend)
#     print(model_wrapped.model.qconfig)
#
#     torch.quantization.prepare_qat(model_wrapped.model, inplace=True)
#
# def _qat_convert(model_wrapped: ModelWrapper):
#     model_wrapped.model.to('cpu')
#     # Convert to quantized model
#     torch.quantization.convert(model_wrapped.model, inplace=True)
#     model_wrapped.model.eval()