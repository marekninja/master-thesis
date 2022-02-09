from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from enmt import QatTrainer


class RobustCallback(TrainerCallback):
    def __init__(self, trainer: QatTrainer, other_dataset: Dataset, metric_key_prefix: str = "custom_eval", num_beams: int = 1):
        """
        Callback to achieve evaluation on more datasets to help evaluate model robustness during training
        Additional parameters added to TrainerCallback

        Should be used with TrainingArguments:
            evaluation_strategy: "steps": Evaluation is done (and logged) every eval_steps
            eval_steps (int, optional) — Number of update steps between two evaluations if evaluation_strategy="steps". Will default to the same value as logging_steps if not set.
            logging_steps (int, optional, defaults to 500) — Number of update steps between two logs if logging_strategy="steps"


        Args:
            trainer: QatTrainer used to train model
            other_dataset: torch.utils.data.Dataset to evaluate on
            metric_key_prefix: name of dataset/metric to be seen in logs
        """
        super().__init__()
        self.trainer = trainer
        self.other_dataset = other_dataset
        self.metric_key_prefix = metric_key_prefix
        self.num_beams = num_beams

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print("Eval callback:",self.metric_key_prefix)
        self.trainer.custom_evaluate(self.other_dataset, metric_key_prefix=self.metric_key_prefix, num_beams=self.num_beams)
