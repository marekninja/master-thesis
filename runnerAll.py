import comet_ml

from enmt.datasets import EuroParl

from enmt.model_wrapper import ModelWrapper
from enmt.results import Pipeline, Scenario
from copy import deepcopy

modelFP = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
modelDQ = deepcopy(modelFP)
modelSQ = deepcopy(modelFP)
modelQAT = deepcopy(modelFP)


eval = EuroParl(test_size=0.00005,seed=42)


# training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 0.002, 'per_device_train_batch_size': 2,
#                      'per_device_eval_batch_size': 15, 'weight_decay': 0.01, 'save_total_limit': 3,
#                      'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': False,
#                      'fp16': False, 'push_to_hub': False, 'bn_freeze':100000, 'qpar_freeze':120000,
#                       'disable_tqdm':True
#                       }
modelFP.model.to('cpu')
training_argsEval = {'no_cuda': True,'fp16': False,'per_device_eval_batch_size': 4, 'predict_with_generate': True}
pipeEval = Pipeline(Scenario.EVAL, model=modelFP, dataset_eval=eval,
                    training_args=training_argsEval)
print("BLEU on FP cpu")
pipeEval.run()
print()

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

train = EuroParl(test_size=0.95,seed=42)
print(train['train'])
training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 2e-5, 'per_device_train_batch_size': 4,
                     'per_device_eval_batch_size': 4, 'weight_decay': 0.01, 'save_total_limit': 3,
                     'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': False,
                     'fp16': False, 'push_to_hub': False, 'bn_freeze':15000, 'qpar_freeze':22000,
                      'disable_tqdm':False
                     }
pipeEval = Pipeline(Scenario.QUANT_AWARE_TUNE, model=modelQAT, dataset_train=train,
                    training_args=training_argsEval)
print("Calibration:")
pipeEval.run()

modelQAT.quantizeQATConvert()
training_argsEval = {'no_cuda': True,'fp16': False,'per_device_eval_batch_size': 4, 'predict_with_generate': True}
pipeEval = Pipeline(Scenario.EVAL, model=modelQAT, dataset_eval=eval,
                    training_args=training_argsEval)
print("BLEU on QAT cpu")
pipeEval.run()