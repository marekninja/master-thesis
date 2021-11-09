from enmt.datasets.opus100 import Opus100
from enmt.model import ModelWrapper
from enmt.quant_helper import QuantizationMode
from enmt.results import Dataset, Pipeline, Scenario

model = ModelWrapper("Helsinki-NLP/opus-mt-en-sk")
# eval full prec model
# print(model.get_quantized(QuantizationMode.DYNAMIC))


eval = Opus100()

training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 5,
                 'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}

# model.quantize(QuantizationMode.DYNAMIC)
pipe = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
                training_args=training_args)

pipe.run()

# model.quantize(QuantizationMode.DYNAMIC)


# training_args = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 5,
#                  'weight_decay': 0.01, 'save_total_limit': 3, 'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True, 'fp16': False, 'push_to_hub': False}

# pipe = Pipeline(Scenario.EVAL, model=model,
#                 dataset_eval=eval, training_args=training_args)

# pipe.run()

# eval quant model
# compare
