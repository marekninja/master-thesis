import comet_ml
from transformers import EarlyStoppingCallback

from enmt import RobustCallback, CometOneExperimentCallback, CometContinueExperimentCallback, \
    TestRobustCallback
from enmt.datasets import EuroParl, OpenSubtitles

from enmt.model_wrapper import ModelWrapper, _test_translation
from enmt.results import Pipeline, Scenario
from copy import deepcopy

import os
from glob import glob
import re

"""
Comparison of speed 

This experiment measures speed of evaluation QUANTIZATION AWARE TRAINED model trained on EuParl
    using checkpoint 70k of model https://www.comet.ml/marekninja/huggingface/46f1064a08c04f72b8bf54f400bc68b4
    trained for 2 epochs on EuParl, LR 2e-4, effective batch size 64

Compare speed Dynamic Quantization VS. QAT VS. Cpu FP Vs. Cuda FP

Training dataset: Euro Parlament en-sk
Evaluation Euro Parl, Open Subs


metric_key_prefix format:
    trainEuParlFP_EuParl_test_cpu


        model specification:
            scenario of model - train
            dataset of model - Euparl
            train mode - FP
        current run specification:
            current dataset - EuParl
            current scenario - test
            device - cpu  
"""

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



# smaller validation set - to allow for frequent metrics evalation
test_size = 40000
valid_size = 0
batch_size = 16
valid_batch_size = batch_size
eval_batch_size_gpu = batch_size
eval_batch_size_cpu = batch_size * 2 # 32 to be same as other experiments
grad_acc_steps = 4
train_epochs = 2 # overiden by max_steps
warmup_steps = 0
eval_steps = 200
# max_steps = 125000# 250k update steps maximum, overides train_epochs...
max_steps = -1 # is negative => is not used; otherwise overides train_epochs
save_total_limit = 3
bn_freeze = int(
    round((639158 / 64) * (3/8)))  # 2/3 of all global steps, based on Pytorch tutorial should be bigger ten qpar_freeze
qpar_freeze = int(round((639158 / 64)* 0.25))  # 1/2 of all global steps

saved_model = "./saved_models/trained/FP_marian_3_marianmt_v2_en-sk_openSubs-euparl_model"
saved_tokenizer = "./saved_models/trained/FP_marian_3_marianmt_v2_en-sk_openSubs-euparl_tokenizer"
# tokenizer does not have state...
experiment_name = "QAT_CPU_EuParl measureSpeed"

# test_size = 0.99995
# test_size = 0.999
# valid_size = 40
# batch_size = 2
# valid_batch_size = 2 * batch_size
# eval_batch_size_gpu = 2 * batch_size
# eval_batch_size_cpu = batch_size // 2
# grad_acc_steps = 1
# train_epochs = 2
# steps = (8000 * train_epochs) // (batch_size * grad_acc_steps)
# bn_freeze = int(round(steps*0.5)) # 1/2 of all global steps
# qpar_freeze = int(round(steps*(2/3))) # 2/3 of all global steps



training_argsCalibrate = {'no_cuda': True, 'fp16': False, 'per_device_eval_batch_size': eval_batch_size_cpu,
                     'predict_with_generate': True,
                     "report_to": "none"
                     }
modelWrapped = ModelWrapper(pretrained_model_name_or_path=saved_model, pretrained_tokenizer_name_or_path=saved_tokenizer)


# 2. Calibrate on EuParl train set
modelWrapped.quantizeStaticStart(test_tr=True)
calibration = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipe = Pipeline(Scenario.CALIBRATE, model=modelWrapped, dataset=calibration,
                training_args=training_argsCalibrate)

callback5 = CometOneExperimentCallback()
pipe.trainer.add_callback(callback5)

print("Calibrate Static Quantization on EuroParl (model previously pre-trained FP) :")
pipe.run()

_test_translation(modelWrapped)

modelWrapped.model.save_pretrained('./saved_models/trainedStaticQ/SQ_FP_marian_3_marianmt_v2_en-sk_euparl-openSubs_model',
                                   push_to_hub=False)
modelWrapped.tokenizer.save_pretrained('./saved_models/trainedStaticQ/SQ_FP_marian_3_marianmt_v2_en-sk_euparl-openSubs_model',
                                       push_to_hub=False)




modelWrapped.quantizeQATConvert()

modelSize = modelWrapped.getSize()
print("Size of model state_dict on disk", modelSize)


training_argsEval = {'no_cuda': True, 'fp16': False, 'per_device_eval_batch_size': eval_batch_size_cpu,
                     'predict_with_generate': True,
                     "report_to": "none"
                     }
train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipeEval = Pipeline(Scenario.EVAL, model=modelWrapped, dataset=train,
                    training_args=training_argsEval, metric_key_prefix="compare_speed_EuParl_test")
pipeEval.trainer.add_callback(CometContinueExperimentCallback())
print("BLEU in-domain (EuParl) on QAT cpu")
pipeEval.run()


training_argsEval = {'no_cuda': True, 'fp16': False, 'per_device_eval_batch_size': eval_batch_size_cpu,
                     'predict_with_generate': True,
                     "report_to": "none"
                     }
validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
pipeEval = Pipeline(Scenario.EVAL, model=modelWrapped, dataset=validation,
                    training_args=training_argsEval, metric_key_prefix="compare_speed_OpenSubs_test")
pipeEval.trainer.add_callback(CometContinueExperimentCallback())
print("BLEU out-of-domain (OpenSubs) on QAT cpu")
pipeEval.run()

comet_ml.get_global_experiment().log_metric("size_on_disk", modelSize)
comet_ml.get_global_experiment().set_name(experiment_name)
comet_ml.get_global_experiment().end()