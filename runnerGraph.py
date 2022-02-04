from enmt.datasets import Opus100
from enmt.datasets import OpenSubtitles
from enmt.datasets import Ubuntu
from enmt.datasets import EuroParl

from enmt.model_wrapper import ModelWrapper
from enmt.quant_helper import QuantizationMode
from enmt.results import Pipeline, Scenario

import torch
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import copy


model = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
print(f" Before quant, size: {model.getSize()}")

quantized_model = copy.deepcopy(model)

quantized_model.model.to('cpu')
quantized_model.model.eval()

# Fuse Conv, bn and relu
# model.model.fuse_model()

num_calibration_batches = 32

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}
prepared_model = prepare_fx(quantized_model.model, qconfig_dict)

# print(prepared_model.graph)
# Calibrate first
print('FXGraph Post Training Quantization Prepare: Inserting Observers')
# print('\n Inverted Residual Block:After observer insertion \n\n', model.model.features[1].conv)

quantized_model.model = prepared_model
# Calibrate with the training set
calibrate = EuroParl(test_size=0.00001)
training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 5,
                 'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}
pipe = Pipeline(Scenario.EVAL, model=quantized_model, dataset_eval=calibrate,
                training_args=training_args)
pipe.run()

# evaluate(model.model, criterion, data_loader, neval_batches=num_calibration_batches)

print('Post Training Quantization: Calibration done')

# Convert to quantized model
quantized_model.model = convert_fx(prepared_model)
print(quantized_model.model)
print('Post Training Quantization: Convert done')
# print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
#       model.model.features[1].conv)

print("Size of model after quantization")
print("Size of model after quantization",quantized_model.getSize())

# top1, top5 = evaluate(model.model, criterion, data_loader_test, neval_batches=num_eval_batches)
# print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

eval = EuroParl(test_size=0.00010)


training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 5,
                 'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}

pipeEval = Pipeline(Scenario.EVAL, model=quantized_model, dataset_eval=eval,
                training_args=training_argsEval)
pipeEval.run()


