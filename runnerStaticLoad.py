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

tokenized = model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True)
# prepared = model.model.prepare_decoder_input_ids_from_labels(labels=tokenized.data["input_ids"])
#
# translated = model.model.forward(input_ids = tokenized.data["input_ids"],
#                                  attention_mask=tokenized.data['attention_mask'],
#                                  decoder_input_ids = prepared)
# try:
#     translated = torch.argmax(translated.logits,-1)
# except:
#     translated = np.argmax(translated.logits, -1)

translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])



# model.model.config.use_cache = False

model.model.to('cpu')
model.model.eval()


# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.model.qconfig)

torch.quantization.prepare(model.model, inplace=True)
# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')


# Convert to quantized model
torch.quantization.convert(model.model, inplace=True)
print('Post Training Quantization: Convert done')

print("Size of model after quantization",model.getSize())
saved_model_dir = "./saved_models/"
scripted_quantized_model_file = "mariannmt-en-sk-static-quant-noneEmbed-euparl.pth"
model.model.load_state_dict(torch.load(saved_model_dir+scripted_quantized_model_file))
print("Size of model after load",model.getSize())

eval = EuroParl(test_size=0.00001,seed=42)


model.model.config.use_cache = True

training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4,
                     'per_device_eval_batch_size': 15, 'weight_decay': 0.01, 'save_total_limit': 3,
                     'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True,
                     'fp16': False, 'push_to_hub': False}

pipeEval = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
                training_args=training_argsEval)
pipeEval.run()

tokenized = model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True)
# prepared = model.model.prepare_decoder_input_ids_from_labels(labels=tokenized.data["input_ids"])
#
# translated = model.model.forward(input_ids = tokenized.data["input_ids"])
translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])