from enmt.datasets import Opus100
from enmt.datasets import OpenSubtitles
from enmt.datasets import Ubuntu
from enmt.datasets import EuroParl

from enmt.model_wrapper import ModelWrapper
from enmt.quant_helper import QuantizationMode
from enmt.results import Pipeline, Scenario

import comet_ml

import torch
# COMET_API_KEY=kOsVFPPIeH1LFMpo1NeuG5QrT
# model = ModelWrapper("Helsinki-NLP/opus-mt-en-sk")
# # eval full prec model
# # print(model.get_quantized(QuantizationMode.DYNAMIC))


# eval = Opus100()

# training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 5,
#                  'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}

# # model.quantize(QuantizationMode.DYNAMIC)
# pipe = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
#                 training_args=training_args)

# pipe.run()


model = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
print(f" Before quant, size: {model.getSize()}")

eval = EuroParl(test_size=0.00001)


training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4,
                 'per_device_eval_batch_size': 1,'weight_decay': 0.01, 'save_total_limit': 3,
                 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True,
                 'fp16': False, 'push_to_hub': False}


# model.quantizeDynamic(test_tr=False)
translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])

model.model.to('cpu')
model.model = torch.quantization.quantize_dynamic(
        model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
    )

translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])


pipe = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
                training_args=training_args)
pipe.run()

translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])

# training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 5,
#                  'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}


# model.quantizeDynamic(QuantizationMode.DYNAMIC)
# print(f" After quant, size: {model.getSize()}")
# pipe = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
#                 training_args=training_args)
# pipe.run()

# model.quantize(QuantizationMode.DYNAMIC)


# training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 5,
#                  'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}

# pipe = Pipeline(Scenario.EVAL, model=model,
#                 dataset_eval=eval, training_args=training_args)

# pipe.run()

# eval quant model
# compare
