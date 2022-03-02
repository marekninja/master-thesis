from enum import Enum
from transformers import Seq2SeqTrainer, TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl
from transformers import Seq2SeqTrainingArguments
from transformers.integrations import CometCallback

from .qat_trainer import QatTrainingArgs, QatTrainer, LogSeq2SeqTrainer
from .dataset import Dataset
from datasets import load_metric

from typing import List, Optional

from transformers.data.data_collator import DataCollatorForSeq2Seq
from enmt.model_wrapper import ModelWrapper
import numpy as np
import torch
import uuid
import os


class Scenario(Enum):
    """Scenarios for Pipeline.  Should cover all the needs of enmt framework

    Args:
        Enum (EVAL): Evaluate provided model
        Enum (TRAIN): Train provided model
        Enum (FT_EVALL: Evaluate model on validation set before fine tuning
        Enum (QUANT_AWARE_TUNE_EVAL): Quantization-Aware fine-tuning of provided model and evaluation

    """
    EVAL = "EVALUATE"
    TRAIN = "TRAIN_EVAL"
    FT_EVAL = "FINE-TUNE_EVAL"
    QUANT_AWARE_TUNE = "QUANT_AWARE_TUNE"  # QAT uses modified training loop
    # QUANT_AWARE_TRAIN_EVAL = "QUANT_AWARE_TRAIN_EVAL"


class Pipeline():
    """Pipeline class, implementing scenarios
    """

    def __init__(self, scenario: Scenario, model: ModelWrapper, dataset: Dataset = None,
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
                                'push_to_hub': False},
                 callbacks: Optional[List[TrainerCallback]] = None,
                 metric_key_prefix: str = "eval"):

        # model_name = model_checkpoint.split("/")[-1]
        self.metric_key_prefix = metric_key_prefix
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.modelWrapper = model
        self.metric = load_metric("sacrebleu")
        self.scenario = scenario

        def name(x: str = "pipeline"): return x + "_" + uuid.uuid4().hex + "_" + self.scenario.value

        if 'output_dir' not in training_args.keys():
            print("Pipeline: output_dir not specified. Generating unique...")
            dir = ""
            while True:
                dir = name()
                if not os.path.isdir(dir):
                    break

            training_args['output_dir'] = dir

        else:
            dir = training_args['output_dir']
            while True:
                if not os.path.isdir(dir):
                    break
                dir = name(training_args['output_dir'])

            training_args['output_dir'] = dir



        print("output_dir for Pipeline is: ", training_args['output_dir'])

        if scenario == Scenario.QUANT_AWARE_TUNE:
            # training_args['evaluation_strategy'] = "no"
            # print("evaluation strategy not supported for QAT, yet...")
            self.training_args = QatTrainingArgs(
                # output_dir=model.pretrained_model_name_or_path + "_" + scenario.value,
                **training_args
            )
        else:
            self.training_args = Seq2SeqTrainingArguments(
                # output_dir=model.pretrained_model_name_or_path + "_" + scenario.value,
                **training_args
            )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model)

        self.config = self.model.config.to_dict()

        if dataset is not None:
            dataset.preprocess(
                tokenizer=self.tokenizer,
                max_input_length=self.config['max_length'], max_target_length=self.config['max_length'], prefix="")

        # if dataset is not None:
        #     dataset.preprocess(
        #         tokenizer=self.tokenizer,
        #         max_input_length=self.config['max_length'], max_target_length=self.config['max_length'], prefix="")

        # if training_args['predict_with_generate'] == True:
        #     compute_metrics =  self._compute_metrics_generate
        # else:
        #     compute_metrics = self._compute_metrics_predict

        if scenario == Scenario.QUANT_AWARE_TUNE or scenario.TRAIN:
            if "train" not in dataset.sets:
                raise RuntimeError("Dataset does not have 'train' split")
            if "val" not in dataset.sets:
                raise RuntimeError("Dataset does not have 'val' split")

        if scenario == Scenario.QUANT_AWARE_TUNE:
            self.trainer = QatTrainer(
                self.model,
                self.training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['val'],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks
            )
        elif scenario == Scenario.TRAIN:
            self.trainer = LogSeq2SeqTrainer(
                self.model,
                self.training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['val'],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks
            )
        elif scenario == Scenario.EVAL:
            self.trainer = LogSeq2SeqTrainer(
                self.model,
                self.training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics,
                callbacks=callbacks
            )
        elif scenario == Scenario.FT_EVAL:
            self.trainer = LogSeq2SeqTrainer(
                self.model,
                self.training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['val'],
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

        if scenario in [Scenario.EVAL, Scenario.FT_EVAL]:
            print(self.trainer.evaluate(metric_key_prefix= self.metric_key_prefix))

        elif scenario == Scenario.QUANT_AWARE_TUNE or scenario == Scenario.TRAIN:

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
