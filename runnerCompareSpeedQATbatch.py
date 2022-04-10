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

import torch
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

CUDA_VISIBLE_DEVICES=5 COMET_API_KEY=apikey python runnerFile.py

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
batch_size = 4
eval_batch_size_cpu = batch_size * 2 # 32 to be same as other experiments



fp_saved_model = './saved_models/trained/FP_marian_6_marianmt_v2_en-sk_euparl-openSubs_model_from_trainer'
qat_saved_model = "/mnt/local/disk1/klasifikace_reflexe/MT_petrovic/in_progress/FP_marian_6_QAT_fine-tuned/75000_FP_marian_6_QAT_find/pytorch_model.bin"
experiment_name = "QAT_CPU_EuParl measureSpeed"

# modelQAT = ModelWrapper(pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
modelWrapped = ModelWrapper(pretrained_model_name_or_path=fp_saved_model)
modelWrapped.quantizeQATStart(test_tr=True)

modelWrapped.model.load_state_dict(torch.load(qat_saved_model),strict=False)

modelWrapped.quantizeQATConvert(test_tr=True)

modelSize = modelWrapped.getSize()
print("Size of model state_dict on disk", modelSize)


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

_test_translation(model_wrapped=modelWrapped)
comet_ml.get_global_experiment().log_metric("size_on_disk",modelSize)
comet_ml.get_global_experiment().log_metric("quantization","QAT")
comet_ml.get_global_experiment().log_metric("quan_specs","embeds in FP")
comet_ml.get_global_experiment().log_metric("device","CPU")
comet_ml.get_global_experiment().log_metric("model_code","modeling_marian_quant_v2")
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