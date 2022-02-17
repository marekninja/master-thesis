from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import CometCallback

from enmt import QatTrainer
import comet_ml
from transformers.utils import logging
import importlib.util
import os

logger = logging.get_logger(__name__)


# comet_ml requires to be imported before any ML frameworks
_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"
if _has_comet:
    try:
        import comet_ml  # noqa: F401

        if hasattr(comet_ml, "config") and comet_ml.config.get_config("comet.api_key"):
            _has_comet = True
        else:
            if os.getenv("COMET_MODE", "").upper() != "DISABLED":
                logger.warning("comet_ml is installed but `COMET_API_KEY` is not set.")
            _has_comet = False
    except (ImportError, ValueError):
        _has_comet = False

class RobustCallback(TrainerCallback):
    def __init__(self, trainer: QatTrainer, other_dataset: Dataset, metric_key_prefix: str = "custom_eval",
                 num_beams: int = 1):
        """
        Callback to achieve evaluation on more datasets to help evaluate model robustness during training
        Additional parameters added to TrainerCallback

        Should be used with TrainingArguments:
            evaluation_strategy: "steps": Evaluation is done (and logged) every eval_steps
            eval_steps (int, optional) — Number of update steps between two evaluations if evaluation_strategy="steps".
                Will default to the same value as logging_steps if not set.
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
        self.trainer.custom_evaluate(self.other_dataset, metric_key_prefix=self.metric_key_prefix,
                                     num_beams=self.num_beams)

class TestRobustCallback(TrainerCallback):
    def __init__(self, trainer: QatTrainer, test_dataset: Dataset, metric_key_prefix: str = "test_custom_eval",
                 num_beams: int = 1):
        """
        Callback to achieve evaluation on more datasets

        Should be used to evaluate on Test set after training
        """
        super().__init__()
        self.trainer = trainer
        self.other_dataset = test_dataset
        self.metric_key_prefix = metric_key_prefix
        self.num_beams = num_beams

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print("Eval callback:",self.metric_key_prefix)
        self.trainer.custom_evaluate(self.other_dataset, metric_key_prefix=self.metric_key_prefix,
                                     num_beams=self.num_beams)


class QuantizedEvalCallback(TrainerCallback):
    def __init__(self, trainer: QatTrainer, other_dataset: Dataset, metric_key_prefix: str = "custom_quant_eval",
                 num_beams: int = 1, on_eval: bool = False, eval_global_step_freq: int = 10000,on_epoch: bool = True):
        """
        Callback to achieve evaluation on more datasets to help evaluate model robustness during training
        Additional parameters added to TrainerCallback

        Should be used with TrainingArguments:
            evaluation_strategy: "steps": Evaluation is done (and logged) every eval_steps
            eval_steps (int, optional) — Number of update steps between two evaluations if evaluation_strategy="steps".
                Will default to the same value as logging_steps if not set.
                logging_steps (int, optional, defaults to 500) — Number of update steps between two logs if logging_strategy="steps"


        Args:
            trainer: QatTrainer used to train model
            other_dataset: torch.utils.data.Dataset to evaluate on
            metric_key_prefix: name of dataset/metric to be seen in logs
        """
        super().__init__()
        self.on_eval = on_eval
        self.eval_global_step_freq = eval_global_step_freq
        self.on_epoch = on_epoch
        self.trainer = trainer
        self.other_dataset = other_dataset
        self.metric_key_prefix = metric_key_prefix
        self.num_beams = num_beams

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print("Eval callback:",self.metric_key_prefix)

        if self.on_epoch:
            print("quantized callback: epoch")
            metrics= self.trainer.quant_evaluate(self.other_dataset, metric_key_prefix=self.metric_key_prefix,
                                    num_beams=self.num_beams)
            print(self.metric_key_prefix, " : ", metrics)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print("Eval callback:",self.metric_key_prefix)

        if self.on_eval and state.global_step % self.eval_global_step_freq == 0:
            print("quantized callback: on eval")
            metrics = self.trainer.quant_evaluate(self.other_dataset, metric_key_prefix=self.metric_key_prefix,
                                        num_beams=self.num_beams)
            print(self.metric_key_prefix, " : ", metrics)

class CometOneExperimentCallback(CometCallback):
    """Disables end of experiment on end of training"""

    def on_train_end(self, args, state, control, **kwargs):
        # pass
        if self._initialized and state.is_world_process_zero:
            experiment = comet_ml.config.get_global_experiment()
            if (experiment is not None) and (self._log_assets is True):
                logger.info("Logging checkpoints. This may take time.")
                experiment.log_asset_folder(
                    args.output_dir, recursive=True, log_file_name=True, step=state.global_step
                )
        #     experiment.end()

class CometContinueExperimentCallback(CometCallback):
    def setup(self, args, state, model):
        self._initialized = True
        return

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        return

    def on_train_end(self, args, state, control, **kwargs):
        return