import comet_ml
from transformers import EarlyStoppingCallback

from enmt import RobustCallback, CometOneExperimentCallback, CometContinueExperimentCallback, \
    TestRobustCallback
from enmt.datasets import EuroParl, OpenSubtitles

from enmt.model_wrapper import ModelWrapper, _test_translation
from enmt.results import Pipeline, Scenario
from copy import deepcopy

"""
Training FP model from scratch for reasonable amount of steps
    Using reset_parameters on positional embedings for FP pretraining
    During QAT finetuning using sinusoidal positional embedings

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

# this should be saved without positional embedings - they are static be default
saved_model_dir = './saved_models/trained/FP_marian_EmbedsExplore_WeirdDataset_marianmt_v2_en-sk_euparl-openSubs_model_from_trainer'




test_size = 400
valid_size = 400
batch_size = 16
valid_batch_size = batch_size
eval_batch_size_gpu = batch_size
eval_batch_size_cpu = batch_size // 2
grad_acc_steps = 4
train_epochs = 2 # overiden by max_steps
warmup_steps = 0
eval_steps = 1000
# max_steps = 125000# 250k update steps maximum, overides train_epochs...
max_steps = -1 # is negative => is not used; otherwise overides train_epochs
save_total_limit = 2
bn_freeze = int(
    round((639158 / 64) * (3/8)))  # 2/3 of all global steps, based on Pytorch tutorial should be bigger ten qpar_freeze
qpar_freeze = int(round((639158 / 64)* 0.25)) # 1/2 of all global steps
# checkpoints_dir = "./FP_marian_3/"

experiment_name = "QAfineTune EmbedingsAnomaly WeirdDataset"


training_args = {"save_strategy": "no",
                 'per_device_eval_batch_size': valid_batch_size, 'predict_with_generate': True,
                 'generation_num_beams': 1,
                 'no_cuda': False,
                 'fp16': False, 'push_to_hub': False,
                 'disable_tqdm': True,
                 'report_to': "none"
                 }

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

# loading the model previously trained - positional_embedings are calculated from sinusoid
modelQAT = ModelWrapper(pretrained_model_name_or_path=saved_model_dir)
# _test_translation(modelQAT)

# 1. Evaluate on validation set, to know model performance before finetuning
# MODEL SHOULD HAVE LOW SCORE HERE - BECAUSE OF EMBEDINGS BEING DIFFERENT DURING PRETRAINING (THEY ARE FROM SINUSOID NOW)

# 1.1 Eval EuroParl
train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipePreFTeval = Pipeline(Scenario.FT_EVAL, modelQAT, train, training_args, metric_key_prefix="trainEuParlFP_EuParl_eval")
pipePreFTeval.trainer.add_callback(CometOneExperimentCallback())
pipePreFTeval.run()

comet_ml.get_global_experiment().set_name(experiment_name)

# 1.2 Eval OpenSubs
validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
pipePreFTeval = Pipeline(Scenario.FT_EVAL, modelQAT, validation, training_args, metric_key_prefix="trainEuParlFP_OpenSubs_eval")
pipePreFTeval.trainer.add_callback(CometContinueExperimentCallback())
pipePreFTeval.run()

# 2. Fine-Tune for EuroParl - metric for this pipeline is eval_bleu
# 2.1 validate on EuroParl
modelQAT.quantizeQATStart(test_tr=True)
train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipe = Pipeline(Scenario.QUANT_AWARE_TUNE, model=modelQAT, dataset=train,
                training_args=training_args_q)

# 2.1 validate on OpenSubs
validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
validation.preprocess(tokenizer=modelQAT.tokenizer)

callback1 = RobustCallback(pipe.trainer, validation['val'], "trainEuParlFP_fineTuneEuParlQAT_OpenSubs_eval")

callback2 = TestRobustCallback(pipe.trainer, train['test'], "trainEuParlFP_fineTuneEuParlQAT_EuParl_test")
callback3 = TestRobustCallback(pipe.trainer, validation['test'], "trainEuParlFP_fineTuneEuParlQAT_OpenSubs_test")

callback5 = CometContinueExperimentCallback()

pipe.trainer.add_callback(callback1)
pipe.trainer.add_callback(callback2)
pipe.trainer.add_callback(callback3)
pipe.trainer.add_callback(callback5)

print("FineTuning QAT on EuroParl (model previously pre-trained FP) :")
pipe.run()

# model is QAT finetuned for SINUSOIDAL positional embedings
# pipe.trainer.save_model('./saved_models/trained/FP_marian_QAT_embedsExplore_marianmt_v2_en-sk_euparl-openSubs_model_from_trainer')

_test_translation(modelQAT)

comet_ml.get_global_experiment().end()


# FP FINETUNING WITH SINUSOIDAL EMBEDINGS

test_size = 400
valid_size = 400
batch_size = 16
valid_batch_size = batch_size
eval_batch_size_gpu = batch_size
eval_batch_size_cpu = batch_size // 2
grad_acc_steps = 4
train_epochs = 2 # overiden by max_steps
warmup_steps = 0
eval_steps = 1000
# max_steps = 125000# 250k update steps maximum, overides train_epochs...
max_steps = -1 # is negative => is not used; otherwise overides train_epochs
save_total_limit = 2

experiment_name = "FPfineTune EmbedingsAnomaly WeirdDataset"


training_args = {"save_strategy": "no",
                 'per_device_eval_batch_size': valid_batch_size, 'predict_with_generate': True,
                 'generation_num_beams': 1,
                 'no_cuda': False,
                 'fp16': False, 'push_to_hub': False,
                 'disable_tqdm': True,
                 'report_to': "none"
                 }

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
                   'no_cuda': False,
                   'fp16': False, 'push_to_hub': False,
                   'disable_tqdm': True,
                   # 'resume_from_checkpoint':'',
                   'report_to': "none"
                   }

# loading the model previously trained - positional_embedings are calculated from sinusoid
modelQAT = ModelWrapper(pretrained_model_name_or_path=saved_model_dir)
# _test_translation(modelQAT)

# 1. Evaluate on validation set, to know model performance before finetuning
# MODEL SHOULD HAVE LOW SCORE HERE - BECAUSE OF EMBEDINGS BEING DIFFERENT DURING PRETRAINING (THEY ARE FROM SINUSOID NOW)

# 1.1 Eval EuroParl
train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipePreFTeval = Pipeline(Scenario.FT_EVAL, modelQAT, train, training_args, metric_key_prefix="trainEuParlFP_EuParl_eval")
pipePreFTeval.trainer.add_callback(CometOneExperimentCallback())
pipePreFTeval.run()

comet_ml.get_global_experiment().set_name(experiment_name)

# 1.2 Eval OpenSubs
validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
pipePreFTeval = Pipeline(Scenario.FT_EVAL, modelQAT, validation, training_args, metric_key_prefix="trainEuParlFP_OpenSubs_eval")
pipePreFTeval.trainer.add_callback(CometContinueExperimentCallback())
pipePreFTeval.run()

# 2. Fine-Tune for EuroParl - metric for this pipeline is eval_bleu
# 2.1 validate on EuroParl
train = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
pipe = Pipeline(Scenario.TRAIN, model=modelQAT, dataset=train,
                training_args=training_args_q)

# 2.1 validate on OpenSubs
validation = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)
validation.preprocess(tokenizer=modelQAT.tokenizer)

callback1 = RobustCallback(pipe.trainer, validation['val'], "trainEuParlFP_fineTuneEuParlQAT_OpenSubs_eval")

callback2 = TestRobustCallback(pipe.trainer, train['test'], "trainEuParlFP_fineTuneEuParlQAT_EuParl_test")
callback3 = TestRobustCallback(pipe.trainer, validation['test'], "trainEuParlFP_fineTuneEuParlQAT_OpenSubs_test")

callback5 = CometContinueExperimentCallback()

pipe.trainer.add_callback(callback1)
pipe.trainer.add_callback(callback2)
pipe.trainer.add_callback(callback3)
pipe.trainer.add_callback(callback5)

print("FineTuning QAT on EuroParl (model previously pre-trained FP) :")
pipe.run()

# model is QAT finetuned for SINUSOIDAL positional embedings
# pipe.trainer.save_model('./saved_models/trained/FP_marian_FineTune_embedsExplore_marianmt_v2_en-sk_euparl-openSubs_model_from_trainer')

_test_translation(modelQAT)

comet_ml.get_global_experiment().end()