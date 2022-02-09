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


print("*** Training FP Marian model from scratch ***")
modelFP.reset()

train = EuroParl(test_size=0.1, seed=42)

training_args = {'metric_for_best_model': "eval_bleu", 'greater_is_better': True, "load_best_model_at_end": True,
                     "save_strategy": "steps",
                     'evaluation_strategy': 'steps', "save_steps": 2000, "eval_steps": 2000, 'logging_first_step': True,
                     'learning_rate': 2e-5, 'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 8,
                     'per_device_eval_batch_size': 4, 'weight_decay': 0.01, 'save_total_limit': 3,
                     'num_train_epochs': 8, 'predict_with_generate': True, 'no_cuda': False,
                     'fp16': False, 'push_to_hub': False,
                     'disable_tqdm': True,
                     }
pipe = Pipeline(Scenario.TRAIN, model=modelFP, dataset_train=train, dataset_eval=train,
                    training_args=training_args)

subs = OpenSubtitles(test_size=0.1, seed=42)
subs.preprocess(tokenizer=pipe.tokenizer)

callback1 = RobustCallback(pipe.trainer, subs['test'], "open_subs_eval")
# callback2 = RobustCallback(pipeEval.trainer,eval['test'],"bleu_euparl")
callback3 = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)

pipe.trainer.add_callback(callback1)
# pipeEval.trainer.add_callback(callback2)
pipe.trainer.add_callback(callback3)

print("Training:")
pipe.run()


modelFP.model.save_pretrained('./saved_models/trained/marianmt_v2_FP_en-sk_model',push_to_hub=False)
modelFP.tokenizer.save_pretrained('./saved_models/trained/marianmt_v2_FP_en-sk_tokenizer',push_to_hub=False)


training_argsEval = {'no_cuda': False, 'fp16': False, 'per_device_eval_batch_size': 4, 'predict_with_generate': True}
pipeEval = Pipeline(Scenario.EVAL, model=modelFP, dataset_eval=train,
                    training_args=training_argsEval)
print("BLEU on FP cpu")
pipeEval.run()
