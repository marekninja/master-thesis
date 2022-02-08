import comet_ml
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from enmt.datasets import Opus100
from enmt.datasets import OpenSubtitles
from enmt.datasets import Ubuntu
from enmt.datasets import EuroParl

from enmt.model_wrapper import ModelWrapper
from enmt.quant_helper import QuantizationMode
from enmt.results import Pipeline, Scenario

import torch
import numpy as np


model = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
print(f" Before quant, size: {model.getSize()}")

translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])


# model.model.to('cpu')
# model.model.eval()

model.model.to('cuda')
# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
# model.model.qconfig = torch.quantization.default_qconfig
model.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
print(model.model.qconfig)

torch.quantization.prepare_qat(model.model, inplace=True)
# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')

# print("Size of model after quantization",model.getSize())
# saved_model_dir = "./saved_models/"
# scripted_quantized_model_file = "mariannmt-en-sk-static-quant-noneEmbed-euparl.pth"
# model.model.load_state_dict(torch.load(saved_model_dir+scripted_quantized_model_file))
# print("Size of model after load",model.getSize())

# train = EuroParl(test_size=0.6,seed=42)
train = EuroParl(test_size=0.9999,seed=42)
# train = EuroParl(test_size=0.99995,seed=42)


# model.model.config.use_cache = False

training_argsTrain = {'evaluation_strategy': 'epoch', 'learning_rate': 0.002, 'per_device_train_batch_size': 2,
                     'per_device_eval_batch_size': 15, 'weight_decay': 0.01, 'save_total_limit': 3,
                     'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': False,
                     'fp16': False, 'push_to_hub': False, 'bn_freeze':100000, 'qpar_freeze':120000,
                      'disable_tqdm':True
                      }
# 'resume_from_checkpoint': 'Helsinki-NLP/opus-mt-en-sk_QUANT_AWARE_TUNE\checkpoint-63500'



pipeTrain = Pipeline(Scenario.QUANT_AWARE_TUNE, model=model, dataset_train=train,
                     training_args=training_argsTrain)
pipeTrain.run()

print("QAT: done")

model.model.to('cpu')

# eval = EuroParl(test_size=0.01,seed=42)
eval = EuroParl(test_size=0.00001,seed=42)
# Convert to quantized model
torch.quantization.convert(model.model, inplace=True)
print('Convert done. Running eval...')

model.model.eval()

tokenized = model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True)
translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])
#
# saved_model_dir = "./saved_models/"
# scripted_quantized_model_file = "mariannmt-en-sk-QAT-quant-v2-03-euparl-0.4.pth"
# torch.save(model.model.state_dict(),saved_model_dir+scripted_quantized_model_file)


training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4,
                     'per_device_eval_batch_size': 4, 'weight_decay': 0.01, 'save_total_limit': 3,
                     'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True,
                     'fp16': False, 'push_to_hub': False}

pipeEval = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
                    training_args=training_argsEval)
pipeEval.run()


# tokenized = model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True)
# prepared = model.model.prepare_decoder_input_ids_from_labels(labels=tokenized.data["input_ids"])
#
# # translated = model.model.forward(input_ids = tokenized.data["input_ids"])
# translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
# print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])
#
# saved_model_dir = "./saved_models/"
# scripted_quantized_model_file = "mariannmt-en-sk-QAT-quant-v2-02-euparl.pth"
# # torch.jit.save(torch.jit.script(model.model), saved_model_dir + scripted_quantized_model_file)
# torch.save(model.model.state_dict(),saved_model_dir+scripted_quantized_model_file)