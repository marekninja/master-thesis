import comet_ml

from enum import Enum
from transformers import Seq2SeqTrainer, TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl
from transformers import Seq2SeqTrainingArguments
from transformers.integrations import CometCallback

from .qat_trainer import QatTrainingArgs, QatTrainer
from .dataset import Dataset
from datasets import load_metric

from typing import List, Optional

from transformers.data.data_collator import DataCollatorForSeq2Seq
from enmt.model_wrapper import ModelWrapper
import numpy as np
import torch


class Scenario(Enum):
    """Scenarios for Pipeline.  Should cover all the needs of enmt framework

    Args:
        Enum (EVAL): Evaluate provided model
        #  Enum (TRAIN_EVAL): Train provided model and evaluate
        Enum (QUANT_AWARE_TUNE_EVAL): Quantization-Aware fine-tuning of provided model and evaluation
        # Enum (QUANT_AWARE_TRAIN_EVAL): Quantization-Aware training from scratch of provided model and evaluation

    """
    EVAL = "evaluate"
    # TRAIN_EVAL = "TRAIN_EVAL"
    QUANT_AWARE_TUNE = "QUANT_AWARE_TUNE" # QAT uses modified training loop
    # QUANT_AWARE_TRAIN_EVAL = "QUANT_AWARE_TRAIN_EVAL"


class Pipeline():
    """Pipeline class, implementing scenarios
    """

    def __init__(self, scenario: Scenario, model: ModelWrapper, dataset_train: Dataset = None,
                 dataset_eval: Dataset = None,
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
                                'push_to_hub': False}, callbacks: Optional[List[TrainerCallback]] = None):

        # model_name = model_checkpoint.split("/")[-1]
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.modelWrapper = model
        self.metric = load_metric("sacrebleu")
        self.scenario = scenario

        if scenario == Scenario.QUANT_AWARE_TUNE:
            # training_args['evaluation_strategy'] = "no"
            # print("evaluation strategy not supported for QAT, yet...")
            self.training_args = QatTrainingArgs(
                output_dir=model.pretrained_model_name_or_path + "_" + scenario.value,
                **training_args
            )
        else:
            self.training_args = Seq2SeqTrainingArguments(
                output_dir=model.pretrained_model_name_or_path + "_" + scenario.value,
                **training_args
            )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)

        self.config = self.model.config.to_dict()

        if dataset_eval is not None:
            dataset_eval.preprocess(
                tokenizer=self.tokenizer,
                max_input_length=self.config['max_length'], max_target_length=self.config['max_length'], prefix="")

        if dataset_train is not None:
            dataset_train.preprocess(
                tokenizer=self.tokenizer,
                max_input_length=self.config['max_length'], max_target_length=self.config['max_length'], prefix="")

        # if training_args['predict_with_generate'] == True:
        #     compute_metrics =  self._compute_metrics_generate
        # else:
        #     compute_metrics = self._compute_metrics_predict

        if scenario == Scenario.QUANT_AWARE_TUNE:
            self.trainer = QatTrainer(
                self.model,
                self.training_args,
                train_dataset=dataset_train['train'] if dataset_train is not None else None,
                eval_dataset=dataset_eval['test'] if dataset_eval is not None else None,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks
            )
        else:
            self.trainer = Seq2SeqTrainer(
                self.model,
                self.training_args,
                train_dataset=dataset_train['train'] if dataset_train is not None else None,
                eval_dataset=dataset_eval['test'] if dataset_eval is not None else None,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks
            )
        print(
            f"Pipeline with {model.pretrained_model_name_or_path} ready to run!")

    def run(self):
        if not isinstance(self.scenario, Scenario):
            raise TypeError(
                f"Scenario '{self.scenario}' is not instance of Scenario")

        scenario = self.scenario
        print(f"Pipeline running with {scenario}...")

        if scenario == Scenario.EVAL:
            print(self.trainer.evaluate())

        elif scenario == Scenario.QUANT_AWARE_TUNE:

            resume = False if self.training_args.resume_from_checkpoint is None \
                else self.training_args.resume_from_checkpoint

            print(self.trainer.train(resume_from_checkpoint=resume))
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
        if self.training_args.predict_with_generate == False:
            preds = np.argmax(preds, -1)
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

    def _compute_metrics_predict(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        #     print(preds[0])
        #     preds = torch.argmax(preds, -1)

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





