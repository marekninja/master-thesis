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

This experiment measures speed of evaluation CPU FP model trained on EuParl
    using best model of https://www.comet.ml/marekninja/huggingface/46f1064a08c04f72b8bf54f400bc68b4


Compare speed Dynamic Quantization VS. QAT VS. Cpu FP Vs. Cuda FP

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
test_size = 40000 # eval on 40k examples, to have be more representable, seed is same as was during training (same split)
valid_size = 400
batch_size = 32
valid_batch_size = batch_size
eval_batch_size_gpu = batch_size
eval_batch_size_cpu = batch_size
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
# checkpoints_dir = "./FP_marian_3/"
saved_model = "./saved_models/trained/FP_marian_3_marianmt_v2_en-sk_openSubs-euparl_model"
saved_tokenizer = "./saved_models/trained/FP_marian_3_marianmt_v2_en-sk_openSubs-euparl_tokenizer"
# tokenizer does not have state...
experiment_name = "FP_CPU_EuParl measureSpeed"

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

# train = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)


training_args = {"save_strategy": "no",
                 'per_device_eval_batch_size': valid_batch_size, 'predict_with_generate': True,
                 'generation_num_beams': 1,
                 'no_cuda': True,
                 'fp16': False, 'push_to_hub': False,
                 'disable_tqdm': False,
                 'report_to': "none"
                 }

# modelQAT = ModelWrapper(pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
modelWrapped = ModelWrapper(pretrained_model_name_or_path=saved_model, pretrained_tokenizer_name_or_path=saved_tokenizer)

modelSize = modelWrapped.getSize()
print("Size of model state_dict on disk", modelSize)
# _test_translation(modelQAT)

# 1. Evaluate on test set
test = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipeTest = Pipeline(Scenario.EVAL, modelWrapped, test, training_args, metric_key_prefix="compare_speed_EuParl_test")
pipeTest.trainer.add_callback(CometOneExperimentCallback())
pipeTest.run()

comet_ml.get_global_experiment().log_metric("size_on_disk",modelSize)
comet_ml.get_global_experiment().set_name(experiment_name)

training_args = {"save_strategy": "no",
                 'per_device_eval_batch_size': valid_batch_size, 'predict_with_generate': True,
                 'generation_num_beams': 1,
                 'no_cuda': True,
                 'fp16': False, 'push_to_hub': False,
                 'disable_tqdm': False,
                 'report_to': "none"
                 }
test = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
pipeTest = Pipeline(Scenario.EVAL, modelWrapped, test, training_args, metric_key_prefix="compare_speed_OpenSubs_test")
pipeTest.trainer.add_callback(CometContinueExperimentCallback())
pipeTest.run()
comet_ml.get_global_experiment().end()