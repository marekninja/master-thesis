from eval_nmt.model import ModelWrapper
from eval_nmt.quant_helper import QuantizationMode

model = ModelWrapper("Helsinki-NLP/opus-mt-en-sk")
# eval full prec model
print(model.get_quantized(QuantizationMode.DYNAMIC))
# eval quant model
# compare
