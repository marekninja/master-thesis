import math
import time
from dataclasses import dataclass, field

from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import speed_metrics


@dataclass
class QatTrainingArgs(Seq2SeqTrainingArguments):
    bn_freeze: int = field(default=80,
                           metadata={"help": "Freezing batch norm and variance estimates after n prediction steps"})

    qpar_freeze: int = field(default=160, metadata={"help": "Freezing quantizer parameters after n prediction steps"})


from typing import Any, Dict, Union, Optional, List

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    is_fairscale_available,
)

import torch
from packaging import version
from torch import nn

from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.utils import logging


_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True

if is_datasets_available():
    pass

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_fairscale_available():
    dep_version_check("fairscale")

if is_sagemaker_dp_enabled():
    pass
else:
    pass

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

class QatTrainer(Seq2SeqTrainer):
    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
    #
    #     Works both with or without labels.
    #     """
    #     args = self.args
    #
    #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
    #
    #     # if eval is called w/o train init deepspeed here
    #     if args.deepspeed and not self.deepspeed:
    #
    #         # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
    #         # from the checkpoint eventually
    #         deepspeed_engine, _, _ = deepspeed_init(
    #             self, num_training_steps=0, resume_from_checkpoint=None, inference=True
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #
    #     model = self._wrap_model(self.model, training=False)
    #
    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)
    #
    #     batch_size = dataloader.batch_size
    #
    #     logger.info(f"***** Running {description} *****")
    #     if isinstance(dataloader.dataset, collections.abc.Sized):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")
    #
    #     model.eval()
    #
    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = dataloader.dataset
    #
    #     if is_torch_tpu_available():
    #         dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)
    #
    #     if args.past_index >= 0:
    #         self._past = None
    #
    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     # Will be useful when we have an iterable dataset so don't know its length.
    #
    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #
    #         if step > args.bn_freeze:
    #             self.model.apply(torch.quantization.disable_observer)
    #             print("Observers freezed!")
    #
    #         if step > args.qpar_freeze:
    #             self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    #             print("BN Stats freezed!")
    #
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size
    #
    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #
    #         if is_torch_tpu_available():
    #             xm.mark_step()
    #
    #         # Update containers on host
    #         if loss is not None:
    #             losses = self._nested_gather(loss.repeat(batch_size))
    #             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
    #         if logits is not None:
    #             logits = self._pad_across_processes(logits)
    #             logits = self._nested_gather(logits)
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
    #         if labels is not None:
    #             labels = self._pad_across_processes(labels)
    #             labels = self._nested_gather(labels)
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
    #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
    #
    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )
    #
    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, labels_host = None, None, None
    #
    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")
    #
    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #
    #     # Number of samples
    #     if not isinstance(eval_dataset, IterableDataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         num_samples = observed_num_examples
    #
    #     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    #     # samplers has been rounded to a multiple of batch_size, so we truncate.
    #     if all_losses is not None:
    #         all_losses = all_losses[:num_samples]
    #     if all_preds is not None:
    #         all_preds = nested_truncate(all_preds, num_samples)
    #     if all_labels is not None:
    #         all_labels = nested_truncate(all_labels, num_samples)
    #
    #     # Metrics!
    #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #         metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}
    #
    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)
    #
    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    #
    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
    #
    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.state.global_step > self.args.bn_freeze:
            self.model.apply(torch.quantization.disable_observer)
            # print("Observers freezed!")

        if self.state.global_step > self.args.qpar_freeze:
            self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            # print("BN Stats freezed!")

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def custom_evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "custom_eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Modification of evaluate(), used only for callback to eval on more datasets...
        Change - does not contain call to on_evaluate callbacks (would be infinite...)
        PLEASE SET METRIC_KEY_PREFIX

        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())


        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


class LogSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Only to have custom_evaluate to support eval callback on different datasets

    """


    def custom_evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "custom_eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Modification of evaluate(), used only for callback to eval on more datasets...
        Change - does not contain call to on_evaluate callbacks (would be infinite...)
        PLEASE SET METRIC_KEY_PREFIX

        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())


        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
