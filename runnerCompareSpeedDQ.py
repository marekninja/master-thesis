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

This experiment measures speed of evaluation DYNAMICALY QUANTIZED model trained on EuParl
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

CUDA_VISIBLE_DEVICES=-1 COMET_API_KEY=apikey python runnerFile.py

Profiling:

import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        pipeEval.run()

"""



# smaller validation set - to allow for frequent metrics evalation
test_size = 40000
valid_size = 400
batch_size = 16
eval_batch_size_cpu = batch_size * 2 # 32 to be same as other experiments

fp_saved_model = './saved_models/trained/FP_marian_6_marianmt_v2_en-sk_euparl-openSubs_model_from_trainer'
# tokenizer does not have state...
experiment_name = "DQ_CPU_EuParl measureSpeed"


# train = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)


modelWrapped = ModelWrapper(pretrained_model_name_or_path=saved_model)
modelWrapped.quantizeDynamic(True)

modelSize = modelWrapped.getSize()
print("Size of model state_dict on disk", modelSize)
# _test_translation(modelQAT)


training_args = {"save_strategy": "no",
                 'per_device_eval_batch_size': eval_batch_size_cpu, 'predict_with_generate': True,
                 'generation_num_beams': 1,
                 'no_cuda': True,
                 'fp16': False, 'push_to_hub': False,
                 'disable_tqdm': False,
                 'report_to': "none"
                 }
# 1. Evaluate on test set
test = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipeTest = Pipeline(Scenario.EVAL, modelWrapped, test, training_args, metric_key_prefix="compare_speed_EuParl_test")
pipeTest.trainer.add_callback(CometOneExperimentCallback())
pipeTest.run()

comet_ml.get_global_experiment().log_metric("size_on_disk",modelSize)
comet_ml.get_global_experiment().set_name(experiment_name)

training_args = {"save_strategy": "no",
                 'per_device_eval_batch_size': eval_batch_size_cpu, 'predict_with_generate': True,
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