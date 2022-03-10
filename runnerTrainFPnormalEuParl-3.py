import comet_ml
from transformers import EarlyStoppingCallback

from enmt import RobustCallback, CometOneExperimentCallback, CometContinueExperimentCallback, \
    TestRobustCallback
from enmt.datasets import EuroParl, OpenSubtitles

from enmt.model_wrapper import ModelWrapper, _test_translation
from enmt.results import Pipeline, Scenario
from copy import deepcopy

import torch

"""
Training FP model from scratch for reasonable amount of steps
    Goal is to have trained model on EuroParl and use it for QAT finetuning

Training dataset: Euro Parlament en-sk
Evaluation Euro Parl, Open Subs

Trained model: FP_marian_3_marianmt_v2_en-sk_openSubs-euparl_
                    model and tokenizer

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

modelQAT = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")

_test_translation(modelQAT)

print("*** Training FP Marian model from scratch ***")
modelQAT.reset()
modelQAT.model._keys_to_ignore_on_save = None

_test_translation(modelQAT)

test_size = 400
valid_size = 400
batch_size = 16
valid_batch_size = batch_size
eval_batch_size_gpu = batch_size
eval_batch_size_cpu = batch_size // 2
grad_acc_steps = 4
train_epochs = 10 # overiden by max_steps
warmup_steps = 4000
max_steps = 5003# 250k update steps maximum, overides train_epochs...
save_total_limit = 50
bn_freeze = int(
    round(250e3 * (2 / 3)))  # 2/3 of all global steps, based on Pytorch tutorial should be bigger ten qpar_freeze
qpar_freeze = int(round(250e3* 0.5))  # 1/2 of all global steps
experiment_name = "FP train EuParl normal"

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
train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)

training_args = {'output_dir': "/mnt/local/disk1/klasifikace_reflexe/MT_petrovic/in_progress/FP_marian_5_small",
                 'metric_for_best_model': "eval_loss", 'greater_is_better': False, "load_best_model_at_end": True,
                 "save_strategy": "steps",
                 'evaluation_strategy': 'steps', "save_steps": 5000, "eval_steps": 5000, 'logging_first_step': True,
                 # 'evaluation_strategy': 'steps', "save_steps": 500, "eval_steps": 500, 'logging_first_step': True,
                 'learning_rate': 2e-4, 'per_device_train_batch_size': batch_size, 'warmup_steps': warmup_steps,
                 # 'learning_rate': 2e-5, 'per_device_train_batch_size': batch_size, 'warmup_steps':0,
                 'gradient_accumulation_steps': grad_acc_steps,
                 'per_device_eval_batch_size': valid_batch_size, 'weight_decay': 0.01, 'save_total_limit': save_total_limit,
                 'num_train_epochs': train_epochs, "max_steps": max_steps, 'predict_with_generate': True,
                 'generation_num_beams': 1,
                 # 'bn_freeze': bn_freeze, 'qpar_freeze': qpar_freeze,
                 'no_cuda': False,
                 'fp16': False,
                 # 'hub_model_id':'marekninja/marian_euparl_test',
                 # 'hub_strategy':'every_save',
                 'push_to_hub': True,
                 'disable_tqdm': True,
                 # 'resume_from_checkpoint':'/mnt/local/disk1/klasifikace_reflexe/MT_petrovic/in_progress/FP_marian_3/checkpoint-80000',
                 'report_to': "none"
                 }

pipe = Pipeline(Scenario.TRAIN, model=modelQAT, dataset=train,
                training_args=training_args, metric_key_prefix="trainEuParlFP_EuParl_eval")

validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
validation.preprocess(tokenizer=pipe.tokenizer)

callback1 = RobustCallback(pipe.trainer, validation['val'], "trainEuParlFP_OpenSubs_eval")

callback2 = TestRobustCallback(pipe.trainer, train['test'], "trainEuParlFP_EuParl_test_cuda")
callback3 = TestRobustCallback(pipe.trainer, validation['test'], "trainEuParlFP_OpenSubs_test_cuda")

callback4 = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)

callback5 = CometOneExperimentCallback()

pipe.trainer.add_callback(callback1)

pipe.trainer.add_callback(callback2)
pipe.trainer.add_callback(callback3)

pipe.trainer.add_callback(callback4)

pipe.trainer.add_callback(callback5)

print("Training FP on EuroParl:")
pipe.run()

pipe.trainer.push_to_hub()
pipe.trainer.save_model('./saved_models/trained/FP_marian_5_smaller_marianmt_v2_en-sk_euparl-openSubs_model_from_trainer')

_test_translation(modelQAT)

comet_ml.get_global_experiment().set_name(experiment_name)


# torch.save(modelQAT.model.state_dict(),'./saved_models/trained/FP_marian_5_marianmt_v2_en-sk_euparl-openSubs_model/model_state_dict.pth')
#
modelQAT.model.save_pretrained('./saved_models/trained/FP_marian_5_smaller_marianmt_v2_en-sk_euparl-openSubs_model',
                               push_to_hub=False)
# modelQAT.tokenizer.save_pretrained('./saved_models/trained/FP_marian_5_marianmt_v2_en-sk_euparl-openSubs_tokenizer',
#                                    push_to_hub=False)

# train = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
# validation = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
# modelQAT.quantizeQATConvert()
#
# training_argsEval = {'no_cuda': True, 'fp16': False, 'per_device_eval_batch_size': eval_batch_size_cpu,
#                      'predict_with_generate': True,
#                      "report_to": "none"
#                      }
# pipeEval = Pipeline(Scenario.EVAL, model=modelQAT, dataset=train,
#                     training_args=training_argsEval, metric_key_prefix="trainEuParlQAT_EuParl_test_cpu")
# pipeEval.trainer.add_callback(CometContinueExperimentCallback())
# print("BLEU in-domain (EuParl) on QAT cpu")
# pipeEval.run()
#
# pipeEval = Pipeline(Scenario.EVAL, model=modelQAT, dataset=validation,
#                     training_args=training_argsEval, metric_key_prefix="trainEuParlQAT_OpenSubs_test_cpu")
# pipeEval.trainer.add_callback(CometContinueExperimentCallback())
# print("BLEU out-of-domain (OpenSubs) on QAT cpu")
# pipeEval.run()

comet_ml.get_global_experiment().end()
