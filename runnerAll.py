import comet_ml
from transformers import EarlyStoppingCallback

from enmt import RobustCallback
from enmt.datasets import EuroParl, OpenSubtitles

from enmt.model_wrapper import ModelWrapper
from enmt.results import Pipeline, Scenario
from copy import deepcopy



"""
Running on LINUX

nvidia-smi -L
    lists available cuda devices on system
    use the number (might be different indexing)

CUDA_VISIBLE_DEVICES=5 python runnerAll.py 

Profiling:

import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        pipeEval.run()

"""

modelFP = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
# modelDQ = deepcopy(modelFP)
# modelSQ = deepcopy(modelFP)
modelQAT = deepcopy(modelFP)

eval = EuroParl(test_size=0.00005, seed=42)

# training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 0.002, 'per_device_train_batch_size': 2,
#                      'per_device_eval_batch_size': 15, 'weight_decay': 0.01, 'save_total_limit': 3,
#                      'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': False,
#                      'fp16': False, 'push_to_hub': False, 'bn_freeze':100000, 'qpar_freeze':120000,
#                       'disable_tqdm':True
#                       }
# modelFP.model.to('cpu')
# training_argsEval = {'no_cuda': True,'fp16': False,'per_device_eval_batch_size': 4, 'predict_with_generate': True}
# pipeEval = Pipeline(Scenario.EVAL, model=modelFP, dataset_eval=eval,
#                     training_args=training_argsEval)
# print("BLEU on FP cpu")
# pipeEval.run()
# print()

# print("*** DYNAMIC QUANTIZATION ***")
# modelDQ.model.to('cpu')
# modelDQ.quantizeDynamic()
# training_argsEval = {'no_cuda': True,'fp16': False,'per_device_eval_batch_size': 4, 'predict_with_generate': True}
# modelDQ.model.to('cpu')
# modelDQ.model.eval()
# pipeEval = Pipeline(Scenario.EVAL, model=modelDQ, dataset_eval=eval,
#                     training_args=training_argsEval)
# print("BLEU on DQ cpu")
# pipeEval.run()
# print()


# print("*** STATIC QUANTIZATION ***")
# modelSQ.quantizeStaticStart()
# calibrate = EuroParl(test_size=0.00005,seed=42)
# training_argsEval = {'no_cuda': True,'fp16': False,'per_device_eval_batch_size': 4, 'predict_with_generate': True}
# pipeEval = Pipeline(Scenario.EVAL, model=modelSQ, dataset_eval=calibrate,
#                     training_args=training_argsEval)
# print("Calibration:")
# pipeEval.run()
#
# modelSQ.quantizeStaticConvert()
# training_argsEval = {'no_cuda': True,'fp16': False,'per_device_eval_batch_size': 4, 'predict_with_generate': True}
# pipeEval = Pipeline(Scenario.EVAL, model=modelSQ, dataset_eval=eval,
#                     training_args=training_argsEval)
# print("BLEU on SQ cpu")
# pipeEval.run()
# print()


print("*** Quantization Aware Fine-Tuning ***")
modelQAT.quantizeQATStart()

train = EuroParl(test_size=0.95, seed=42)
print(train['train'])
# training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 2e-5, 'per_device_train_batch_size': 4,
#                      'per_device_eval_batch_size': 4, 'weight_decay': 0.01, 'save_total_limit': 3,
#                      'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': False,
#                      'fp16': False, 'push_to_hub': False, 'bn_freeze':15000, 'qpar_freeze':22000,
#                       'disable_tqdm':False,
#                      }

# to use early stopping:
# 'metric_for_best_model':"eval_bleu", 'greater_is_better':True, load_best_model_at_end:True, "save_strategy": "steps",
# 'evaluation_strategy': 'steps', save_steps: 400, "eval_steps": 200
training_argsEval = {'evaluation_strategy': 'steps', 'eval_steps': 200, 'logging_first_step': True,
                     'learning_rate': 2e-5, 'per_device_train_batch_size': 4,
                     'per_device_eval_batch_size': 4, 'weight_decay': 0.01, 'save_total_limit': 3,
                     'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': False,
                     'fp16': False, 'push_to_hub': False, 'bn_freeze': 15000, 'qpar_freeze': 22000,
                     'disable_tqdm': True,
                     }

pipeEval = Pipeline(Scenario.QUANT_AWARE_TUNE, model=modelQAT, dataset_train=train, dataset_eval=eval,
                    training_args=training_argsEval)
subs = OpenSubtitles(test_size=0.00005, seed=42)
subs.preprocess(tokenizer=pipeEval.tokenizer)

eval.preprocess(pipeEval.tokenizer)

callback1 = RobustCallback(pipeEval.trainer, subs['test'], "open_subs_eval")
# callback2 = RobustCallback(pipeEval.trainer,eval['test'],"bleu_euparl")
# callback3 = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)

pipeEval.trainer.add_callback(callback1)
# pipeEval.trainer.add_callback(callback2)
# pipeEval.trainer.add_callback(callback3)

print("Calibration:")
pipeEval.run()

modelQAT.quantizeQATConvert()
training_argsEval = {'no_cuda': True, 'fp16': False, 'per_device_eval_batch_size': 4, 'predict_with_generate': True}
pipeEval = Pipeline(Scenario.EVAL, model=modelQAT, dataset_eval=eval,
                    training_args=training_argsEval)
print("BLEU on QAT cpu")
pipeEval.run()
