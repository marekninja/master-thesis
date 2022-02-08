import comet_ml

from enmt.datasets import OpenSubtitles
from enmt.datasets import EuroParl

from enmt.model_wrapper import ModelWrapper
from enmt.quant_helper import QuantizationMode
from enmt.results import Pipeline, Scenario

# COMET_API_KEY=kOsVFPPIeH1LFMpo1NeuG5QrT

model = ModelWrapper(
    pretrained_model_name_or_path="Helsinki-NLP/opus-mt-en-sk")
print(f" Before quant, size: {model.getSize()}")

model.model.to('cpu')

eval = EuroParl(test_size=0.01,seed=42)
training_argsEval = {'evaluation_strategy': 'epoch', 'learning_rate': 0.00002, 'per_device_train_batch_size': 4,
                     'per_device_eval_batch_size': 4, 'weight_decay': 0.01, 'save_total_limit': 3,
                     'num_train_epochs': 1, 'predict_with_generate': True, 'no_cuda': True,
                     'fp16': False, 'push_to_hub': False}

pipe = Pipeline(Scenario.EVAL, model=model, dataset_eval=eval,
                training_args=training_argsEval)
pipe.run()


