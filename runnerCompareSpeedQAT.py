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
# checkpoints_dir = "./FP_marian_3/"
checkpoints_dir = "/mnt/local/disk1/klasifikace_reflexe/MT_petrovic/in_progress/FP_marian_3/"
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

# train = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)


dirs = sorted([f for f in glob(os.path.join(checkpoints_dir,"checkpoint-*"))], key= lambda x: int(re.findall(".*checkpoint-(\d+)",x)[0]))
dirs = dirs[1::2]
checkpoints = [c for d in dirs if os.path.isfile(c := os.path.join(d,"pytorch_model.bin"))]

dirs = list(filter(lambda x: int(re.findall(".*checkpoint-(\d+)",x)[0]) == 70000, dirs))

print(dirs)
print(checkpoints)
# exit()

for dir in dirs:


    start_step = re.findall(".*checkpoint-(\d+)",dir)[0]

    training_args_q = {"save_strategy": "no",
                       'evaluation_strategy': 'steps', "eval_steps": eval_steps, 'logging_first_step': True,
                       # 'evaluation_strategy': 'steps', "save_steps": 500, "eval_steps": 500, 'logging_first_step': True,
                       'learning_rate': 2e-4, 'per_device_train_batch_size': batch_size, 'warmup_steps': warmup_steps,
                       # 'learning_rate': 2e-5, 'per_device_train_batch_size': batch_size, 'warmup_steps':0,
                       'gradient_accumulation_steps': grad_acc_steps,
                       'per_device_eval_batch_size': valid_batch_size, 'weight_decay': 0.01,
                       'save_total_limit': save_total_limit,
                       'num_train_epochs': train_epochs, "max_steps": max_steps, 'predict_with_generate': True,
                       'generation_num_beams': 1,
                       'bn_freeze': bn_freeze, 'qpar_freeze': qpar_freeze,
                       'no_cuda': False,
                       'fp16': False, 'push_to_hub': False,
                       'disable_tqdm': True,
                       # 'resume_from_checkpoint':'',
                       'report_to': "none"
                       }


    modelWrapped = ModelWrapper(pretrained_model_name_or_path=dir)


    # 2. Fine-Tune for EuroParl - metric for this pipeline is eval_bleu
    # 2.1 validate on EuroParl
    modelWrapped.quantizeQATStart(test_tr=True)
    train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
    pipe = Pipeline(Scenario.QUANT_AWARE_TUNE, model=modelWrapped, dataset=train,
                    training_args=training_args_q)

    # 2.1 validate on OpenSubs
    validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
    validation.preprocess(tokenizer=modelWrapped.tokenizer)

    callback1 = RobustCallback(pipe.trainer, validation['val'], "trainEuParlFP_fineTuneEuParlQAT_OpenSubs_eval")
    callback5 = CometOneExperimentCallback()

    pipe.trainer.add_callback(callback1)
    pipe.trainer.add_callback(callback5)

    print("FineTuning QAT on EuroParl (model previously pre-trained FP) :")
    pipe.run()

    _test_translation(modelWrapped)

    modelWrapped.model.save_pretrained('./saved_models/trainedQAT/' + start_step + '_FP_marian_3_marianmt_v2_en-sk_euparl-openSubs_model',
                                       push_to_hub=False)
    modelWrapped.tokenizer.save_pretrained('./saved_models/trainedQAT/' + start_step + '_FP_marian_3_marianmt_v2_en-sk_euparl-openSubs_tokenizer',
                                           push_to_hub=False)

    train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
    validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)

    modelWrapped.quantizeQATConvert()

    modelSize = modelWrapped.getSize()
    print("Size of model state_dict on disk", modelSize)

    comet_ml.get_global_experiment().log_metric("size_on_disk", modelSize)
    comet_ml.get_global_experiment().set_name(experiment_name)

    training_argsEval = {'no_cuda': True, 'fp16': False, 'per_device_eval_batch_size': eval_batch_size_cpu,
                         'predict_with_generate': True,
                         "report_to": "none"
                         }
    pipeEval = Pipeline(Scenario.EVAL, model=modelWrapped, dataset=train,
                        training_args=training_argsEval, metric_key_prefix="compare_speed_EuParl_test")
    pipeEval.trainer.add_callback(CometContinueExperimentCallback())
    print("BLEU in-domain (EuParl) on QAT cpu")
    pipeEval.run()

    pipeEval = Pipeline(Scenario.EVAL, model=modelWrapped, dataset=validation,
                        training_args=training_argsEval, metric_key_prefix="compare_speed_OpenSubs_test")
    pipeEval.trainer.add_callback(CometContinueExperimentCallback())
    print("BLEU out-of-domain (OpenSubs) on QAT cpu")
    pipeEval.run()


    comet_ml.get_global_experiment().end()