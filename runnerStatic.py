from enmt.datasets import Opus100
from enmt.datasets import OpenSubtitles
from enmt.datasets import Ubuntu
from enmt.datasets import EuroParl

from enmt.model_wrapper import ModelWrapper
from enmt.quant_helper import QuantizationMode
from enmt.results import Pipeline, Scenario

import torch



model = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
print(f" Before quant, size: {model.getSize()}")

translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])

# model.model.config.use_cache = False

# model.model.to('cpu')
model.model.eval()

# Fuse Conv, bn and relu
# model.model.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
# model.model.qconfig = torch.quantization.default_qconfig
model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.model.qconfig)

torch.quantization.prepare(model.model, inplace=True)
# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
# Calibrate with the training set
# calibrate = EuroParl(test_size=0.995,seed=1)
# calibrate = EuroParl(test_size=0.005,seed=1)
calibrate = EuroParl(test_size=0.00005,seed=42)
training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 4,
                 'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}
pipe = Pipeline(Scenario.EVAL, model=model, dataset_eval=calibrate,
                training_args=training_args)
pipe.run()

# evaluate(model.model, criterion, data_loader, neval_batches=num_calibration_batches)

print('Post Training Quantization: Calibration done')
model.model.to('cpu')
# Convert to quantized model
torch.quantization.convert(model.model, inplace=True)
print('Post Training Quantization: Convert done')
# print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
#       model.model.features[1].conv)

print("Size of model after quantization",model.getSize())

# top1, top5 = evaluate(model.model, criterion, data_loader_test, neval_batches=num_eval_batches)
# print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

eval = EuroParl(test_size=0.00005,seed=42)


training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 4,
                 'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}

pipeEval = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
                training_args=training_argsEval)
pipeEval.run()

translated = model.model.generate(**model.tokenizer("My name is Sarah and I live in London, it is a very nice city", return_tensors="pt", padding=True))
print([model.tokenizer.decode(t, skip_special_tokens=True) for t in translated])
#
# saved_model_dir = "./saved_models/"
# scripted_quantized_model_file = "mariannmt-en-sk-static-v2-euparl.pth"
# # torch.jit.save(torch.jit.script(model.model), saved_model_dir + scripted_quantized_model_file)
# torch.save(model.model.state_dict(),saved_model_dir+scripted_quantized_model_file)