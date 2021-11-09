import comet_ml
from enum import Enum
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers.integrations import CometCallback
from datasets import load_dataset
from datasets import load_metric
from tqdm.auto import tqdm

from typing import List

from transformers.data.data_collator import DataCollatorForSeq2Seq
from enmt.model import ModelWrapper
import numpy as np


class Scenario(Enum):
    """Scenarios for Pipeline.  Should cover all the needs of enmt framework

    Args:
        Enum (EVAL): Evaluate provided model
        Enum (TRAIN_EVAL): Train provided model and evaluate
        Enum (QUANT_AWARE_TUNE_EVAL): Quantization-Aware fine-tuning of provided model and evaluation
        Enum (QUANT_AWARE_TRAIN_EVAL): Quantization-Aware training from scratch of provided model and evaluation

    """
    EVAL = "evaluate"
    TRAIN_EVAL = "TRAIN_EVAL"
    QUANT_AWARE_TUNE_EVAL = "QUANT_AWARE_TUNE_EVAL"
    QUANT_AWARE_TRAIN_EVAL = "QUANT_AWARE_TRAIN_EVAL"


class Dataset():

    def __init__(self, dataset_name, lang1, lang2) -> None:
        self.dataset = self.load(dataset_name, lang1, lang2)
        self.source_lang = lang1
        self.target_lang = lang2
        self.sets = None

    def __getitem__(self, key):
        if self.sets is not None:
            return self.sets[key]
        else:
            return None

    def load(self, dataset_name, lang1, lang2):
        self.dataset = load_dataset(
            dataset_name, lang1=lang1, lang2=lang2)

    def _check_split(self, dataset):
        keys = dataset.keys()

        sets = {}
        for i in ['train', 'validation', 'test']:
            if i in keys:
                sets[i] = dataset[i]
            else:
                sets[i] = None

        if 'train' in keys and 'test' not in keys:
            if 'train' in keys:
                new = dataset['train'].train_test_split(test_size=0.2, seed=1)
                sets['train'] = new['train']
                sets['test'] = new['test']

        if 'train' not in keys:
            raise RuntimeError("Dataset does not have 'train' split")

        self.sets = sets

    def preprocess(self, tokenizer, max_input_length, max_target_length, prefix):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        dataset = self.dataset.map(self._preprocess_function, batched=True)
        self._check_split(dataset)

    def _preprocess_function(self, examples):
        inputs = [self.prefix + ex[self.source_lang]
                  for ex in examples["translation"]]
        targets = [ex[self.target_lang] for ex in examples["translation"]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class Pipeline():
    """Pipeline class, implementig scenarios
    """

    def __init__(self, scenario: Scenario, model: ModelWrapper, dataset_train: Dataset = None, dataset_eval: Dataset = None,
                 training_args={'evaluation_strategy': 'epoch',
                                'learning_rate': 2e-5,
                                'per_device_train_batch_size': 4,
                                'per_device_eval_batch_size': 4,
                                'weight_decay': 0.01,
                                'save_total_limit': 3,
                                'num_train_epochs': 1,
                                'predict_with_generate': True,
                                'no_cuda': True,
                                'fp16': False,
                                'push_to_hub': False}):

        # model_name = model_checkpoint.split("/")[-1]
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.modelWrapper = model
        self.metric = load_metric("sacrebleu")
        self.scenario = scenario

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=model.pretrained_model_name_or_path + "_"+scenario.value,
            **training_args
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)

        config = self.model.config.to_dict()

        dataset_eval.preprocess(
            tokenizer=self.tokenizer,
            max_input_length=config['max_length'], max_target_length=config['max_length'], prefix="")

        self.trainer = Seq2SeqTrainer(
            self.model,
            self.training_args,
            train_dataset=dataset_train['train'] if dataset_train is not None else None,
            eval_dataset=dataset_eval['test'] if dataset_eval is not None else None,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,

        )
        print(
            f"Pipeline with {model.pretrained_model_name_or_path} ready to run!")

    def run(self):
        if not isinstance(self.scenario, Scenario):
            raise TypeError(
                f"Scenario '{self.scenario}' is not instance of Scenario")

        scenario = self.scenario
        print(f"Pipelin running with {scenario}...")
        if scenario == Scenario.EVAL:
            self.trainer.evaluate()

        else:
            raise NotImplementedError()

    # model.to('cpu')

    @staticmethod
    def _postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def _compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
    #     print(preds[0])
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = Pipeline._postprocess_text(
            decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds,
                                     references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(
            pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


class Comparator():
    def __init__(self, results: List[Pipeline]) -> None:
        pass
