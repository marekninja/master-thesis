import comet_ml
from transformers import EarlyStoppingCallback

from enmt import RobustCallback, CometOneExperimentCallback, CometContinueExperimentCallback, \
    TestRobustCallback
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

modelQAT = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")

print("*** Training QAT Marian model from scratch ***")
modelQAT.reset()
modelQAT.quantizeQATStart()

test_size = 40000
valid_size = 40000
batch_size = 32
valid_batch_size = 2 * batch_size
eval_batch_size_gpu = 2 * batch_size
eval_batch_size_cpu = batch_size // 2
grad_acc_steps = 2
train_epochs = 4
warmup_steps = 10000
bn_freeze = int(round(548180*0.5)) # 1/2 of all global steps
qpar_freeze = int(round(548180*(2/3))) # 2/3 of all global steps

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

train = OpenSubtitles(test_size=test_size, valid_size=valid_size, seed=42)


training_args = {'output_dir':"QAT_marian_1",
                 'metric_for_best_model': "eval_bleu", 'greater_is_better': True, "load_best_model_at_end": True,
                 "save_strategy": "steps",
                 'evaluation_strategy': 'steps', "save_steps": 10000, "eval_steps": 10000, 'logging_first_step': True,
                 # 'evaluation_strategy': 'steps', "save_steps": 500, "eval_steps": 500, 'logging_first_step': True,
                 'learning_rate': 2e-4, 'per_device_train_batch_size': batch_size, 'warmup_steps':warmup_steps,
                 # 'learning_rate': 2e-5, 'per_device_train_batch_size': batch_size, 'warmup_steps':0,
                 'gradient_accumulation_steps': grad_acc_steps,
                 'per_device_eval_batch_size': valid_batch_size, 'weight_decay': 0.01, 'save_total_limit': 3,
                 'num_train_epochs': train_epochs, 'predict_with_generate': True, 'generation_num_beams': 1,
                 'bn_freeze': bn_freeze, 'qpar_freeze': qpar_freeze,
                 'no_cuda': False,
                 'fp16': False, 'push_to_hub': False,
                 'disable_tqdm': True,
                 'report_to':"none"
                 }

pipe = Pipeline(Scenario.QUANT_AWARE_TUNE, model=modelQAT, dataset=train,
                training_args=training_args)

euparl = EuroParl(test_size=test_size, valid_size=valid_size, seed=42)
# euparl = EuroParl(test_size=40, valid_size=40, seed=42)
euparl.preprocess(tokenizer=pipe.tokenizer)

small_open = OpenSubtitles(test_size=40, valid_size=40, seed=42)
# small_open.preprocess(tokenizer=pipe.tokenizer)

callback1 = RobustCallback(pipe.trainer, euparl['val'], "euro_parl_eval")

callback2 = TestRobustCallback(pipe.trainer, euparl['test'], "euro_parl_cuda_test")
callback3 = TestRobustCallback(pipe.trainer, train['test'], "open_subs_cuda_test")
# callback3 = TestRobustCallback(pipe.trainer, small_open['test'], "open_subs_cuda_test")

callback4 = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)

callback5 = CometOneExperimentCallback()

pipe.trainer.add_callback(callback1)

pipe.trainer.add_callback(callback2)
pipe.trainer.add_callback(callback3)

pipe.trainer.add_callback(callback4)

pipe.trainer.add_callback(callback5)

print("Training QAT on OpenSubs:")
pipe.run()

modelQAT.model.save_pretrained('./saved_models/trained/QAT_marian_1_marianmt_v2_en-sk_openSubs-euparl_model', push_to_hub=False)
modelQAT.tokenizer.save_pretrained('./saved_models/trained/QAT_marian_1_marianmt_v2_en-sk_openSubs-euparl_tokenizer',
                                   push_to_hub=False)

train = OpenSubtitles(test_size=20, valid_size=valid_size, seed=42)
euparl = EuroParl(test_size=20, valid_size=valid_size, seed=42)
modelQAT.quantizeQATConvert()

training_argsEval = {'no_cuda': True, 'fp16': False, 'per_device_eval_batch_size': eval_batch_size_cpu,
                     'predict_with_generate': True,
                     "report_to": "none"
                     }
pipeEval = Pipeline(Scenario.EVAL, model=modelQAT, dataset=train,
                    training_args=training_argsEval, metric_key_prefix="test_qat_cpu_OpenSubs")
pipeEval.trainer.add_callback(CometContinueExperimentCallback())
print("BLEU in-domain (OpenSubs) on QAT cpu")
pipeEval.run()


pipeEval = Pipeline(Scenario.EVAL, model=modelQAT, dataset=euparl,
                    training_args=training_argsEval, metric_key_prefix="test_qat_cpu_EuroParl")
pipeEval.trainer.add_callback(CometContinueExperimentCallback())
print("BLEU out-of-domain (EuroParl) on QAT cpu")
pipeEval.run()


comet_ml.get_global_experiment().end()