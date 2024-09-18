"""
This module allows to instantiate a model with a wrapped Trainer
"""
import math
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import gc

sys.path.append("../")
sys.path.append("../../")
import collections
from logging import getLogger
from typing import Any, Dict, Optional, Union

import hydra
import torch.distributed as dist
from callbacks import DirManagerCallback, OptunaCallback, PlotCallback
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig
from packaging import version
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from T2 import (
    CamembertT2ForSequenceClassification,
    T2BatchSampler,
    T2Dataset,
    T2PatientBatchSampler,
)
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.file_utils import is_torch_tpu_available
from transformers.integrations import deepspeed_init, hp_params
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    SequentialDistributedSampler,
    ShardSampler,
    get_tpu_sampler,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    has_length,
    seed_worker,
    speed_metrics,
)

# get_model_param_count
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    logging,
)

from constants import COLORS
from data.utils import from_json

TRAINER_STATE_NAME = "trainer_state.json"

log = getLogger(__name__)
logger = logging.get_logger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_accelerate_available():
    from accelerate import skip_first_batches

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


class SaeTrainer(Trainer):
    """
    This class wraps the HuggingFace Trainer.
    """

    def __init__(
        self,
        label2id=None,
        problem_type="multi_label_classification",
        dataset_folder=None,
        n_cr=None,
        nhead=None,
        time_dim=None,
        num_layers=None,
        resume=False,
        label_weights_type="none",
        **kwargs,
    ):
        log.info(f"label_weights_type: {label_weights_type}")
        self.problem_type = problem_type
        self.dataset_folder = dataset_folder
        self.tokenizer = kwargs["tokenizer"]
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()} if label2id is not None else None
        self.n_cr = n_cr
        self.nhead = nhead
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.resume = resume
        self.train_dataset = kwargs["train_dataset"] if "train_dataset" in kwargs else None
        self.num_labels = len(label2id) if label2id is not None else len(kwargs["train_dataset"])
        self.label_weights_type = label_weights_type
        self.backpropagation_schedule = []
        self.acc_loss = None
        self.labels_weight, self.eval_supports = self.get_labels_weight()
        kwargs["model_init"] = self.model_init_regular
        log.info(f"labels_weight: {self.labels_weight} \n eval_supports: {self.eval_supports}")
        kwargs["compute_metrics"] = self.compute_metrics
        # ...
        #     kwargs["model_init"] = self.model_init
        super().__init__(**kwargs)

    @classmethod
    def get_template_trainer(cls, best_model_path):
        config = {
            # "model": "../../models/pretrained/kmembert-base",
            "model": "/home/bourhani@clb.loc/saepred/models/output/optuna/run-31/checkpoint-158920",
            "time_dim": 8,
            "nhead": 8,
            "num_layers": 4,
            # "n_cr": 3,
            "n_cr": 7,
            "label_weights_type": "aggressive",
            "args": {
                # "output_dir": "../../models/output/optuna",
                "output_dir": "../../models/output/dounya",
                "evaluation_strategy": "epoch",
                "logging_strategy": "epoch",
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 32,
                "learning_rate": None,
                "num_train_epochs": 20,
                "save_strategy": "epoch",
                "save_total_limit": 3,  # Earlystopping + 1 to keep best model and allow continuing on checkpoints
                "no_cuda": False,
                "seed": 42,
                "torch_compile": False,
                "fp16": True,
                "fp16_full_eval": False,
                "load_best_model_at_end": True,
                "metric_for_best_model": "macro_avg_aucpr",
                "group_by_length": False,  # faster computations of dynamic padding
                "auto_find_batch_size": False,  # must have installed accelerate
            },
            "data_collator": None,
            # "train_dataset": "../../data/featurized/OncoBERT_8LAB/train",
            "train_dataset": "/home/bourhani@clb.loc/saepred/data/featurized/OncoBERT_nobias_2LAB/train",
            "eval_dataset": None,
            # "dataset_folder": "../../data/featurized/OncoBERT_8LAB",
            "dataset_folder": "/home/bourhani@clb.loc/saepred/data/featurized/OncoBERT_nobias_2LAB",
            # "tokenizer": None,
            "model_init": None,
            "compute_metrics": None,
            "callbacks": [],
            "problem_type": "multi_label_classification",
        }
        config["tokenizer"] = AutoTokenizer.from_pretrained(config["model"])
        label2id = from_json(f"""{config["dataset_folder"]}/label2id.json""")
        config["train_dataset"] = T2Dataset(
            load_from_disk(config["train_dataset"]).rename_column("dt_since_first", "dt")._data, n_cr=config["n_cr"]
        )
        config["model"] = CamembertT2ForSequenceClassification.from_pretrained_body(
            config["model"],
            time_dim=config["time_dim"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            n_cr=config["n_cr"],
            problem_type="multi_label_classification",
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
        )
        config["args"] = TrainingArguments(**config["args"])
        config["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=2),
            OptunaCallback(best_model_path=best_model_path),
            DirManagerCallback(),
        ]
        del config["time_dim"], config["nhead"], config["num_layers"]
        return cls(**config)

    @classmethod
    def from_config(cls, cfg: DictConfig, resume: bool = False):
        cfg["resume"] = resume
        cfg["tokenizer"] = AutoTokenizer.from_pretrained(cfg["model"])
        # ! remove data selection .select(range(100))
        cfg["train_dataset"] = T2Dataset(
            load_from_disk(f"""{cfg["dataset_folder"]}/train""")
            # .select(range(100))
            .rename_column("dt_since_first", "dt")
            ._data,
            n_cr=cfg["n_cr"],
        )
        cfg["eval_dataset"] = T2Dataset(
            load_from_disk(f"""{cfg["dataset_folder"]}/valid""")
            # .select(range(100))
            .rename_column("dt_since_first", "dt")
            ._data,
            n_cr=cfg["n_cr"],
        )
        if "time_dim" in cfg:
            cfg["args"]["group_by_length"] = False  # sequential sampling for T2 is mandatory
        cfg["args"] = TrainingArguments(**cfg["args"])
        cfg["data_collator"] = DataCollatorWithPadding(
            tokenizer=cfg["tokenizer"],
            padding=cfg["padding"],
        )
        cfg["callbacks"] = [PlotCallback(), EarlyStoppingCallback(early_stopping_patience=2)]
        if not cfg["resume"]:
            cfg["callbacks"].append(DirManagerCallback())
        # cfg["model"] = CamembertT2ForSequenceClassification.from_pretrained_body(
        # cfg["model"],
        # time_dim=cfg["time_dim"],
        # nhead=cfg["nhead"],
        # num_layers=cfg["num_layers"],
        # n_cr=cfg["n_cr"],
        # problem_type="multi_label_classification",
        # id2label={v:k for k,v in label2id.items()},
        # label2id=label2id,
        # )
        torch.cuda.manual_seed(cfg["args"].seed)
        torch.manual_seed(cfg["args"].seed)
        torch.cuda.manual_seed_all(cfg["args"].seed)
        torch.backends.cudnn.deterministic = True
        cfg["label2id"] = from_json(f"{cfg['dataset_folder']}/label2id.json")
        del cfg["padding"], cfg["model"]
        return cls(**cfg)

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

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
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

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if has_length(train_dataloader):
            num_update_steps_per_epoch = len(self.backpropagation_schedule)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for _ in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)
        for epoch in range(epochs_trained, num_train_epochs):
            total_batched_samples = 0
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(self.backpropagation_schedule)
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                if step in self.backpropagation_schedule:
                    total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step in self.backpropagation_schedule:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if step in self.backpropagation_schedule:
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (total_batched_samples + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True
            log.info(
                f"total_batched_samples: {total_batched_samples}\nsteps_skipped:{steps_skipped}\nsteps in epoch:{steps_in_epoch}"
            )
            log.info(f"adding: {(total_batched_samples + steps_skipped) / steps_in_epoch} to epoch: {epoch}\n\n")
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

            self.save_state()

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _get_output_dir(self, trial):
        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb

                run_id = wandb.run.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
        return run_dir

    def model_init_regular(self):
        model = CamembertT2ForSequenceClassification.from_pretrained_body(
            self.tokenizer.name_or_path,
            time_dim=self.time_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            n_cr=self.n_cr,
            problem_type="multi_label_classification",
            id2label=self.id2label,
            label2id=self.label2id,
        )
        return model

    def model_init_optuna(self, trial):
        prefix = "../../models/pretrained/"
        dataset_prefix = "../../data/featurized"
        model = trial.suggest_categorical("model", ["OncoBERT_v1.0", "OncoBERT"])
        self.tokenizer = AutoTokenizer.from_pretrained(prefix + model)
        self.data_collator = DataCollatorWithPadding (
            tokenizer=self.tokenizer,
            padding="longest",
        )
        time_dim = trial.suggest_categorical("time_dim", [4, 8, 16, 32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [4, 8, 16])
        if time_dim % nhead != 0:
            raise optuna.TrialPruned()
        num_layers = trial.suggest_int("num_layers", 2, 12)
        n_cr = trial.suggest_int("n_cr", 1, 10)
        label_weights_type = trial.suggest_categorical("label_weights_type", ["None", "aggressive", "max_support"])
        self.label_weights_type = label_weights_type
        self.n_cr = n_cr
        self.dataset_folder = f"{dataset_prefix}/OncoBERT_8LAB"
        # ! remove train for eval dataset
        self.train_dataset = T2Dataset(
            load_from_disk(f"""{self.dataset_folder}/train""").rename_column("dt_since_first", "dt")._data, n_cr=n_cr
        )
        self.eval_dataset = T2Dataset(
            load_from_disk(f"""{self.dataset_folder}/valid""").rename_column("dt_since_first", "dt")._data, n_cr=n_cr
        )
        self.labels_weight, self.eval_supports = self.get_labels_weight()
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        self.args.learning_rate = learning_rate
        log.info(
            f"parameters for this run are: \n model: {model}, \n time_dim: {time_dim}, \n nhead: {nhead}, \n num_layers: {num_layers}, \n n_cr: {n_cr}, \n learning_rate: {learning_rate}"
        )
        self.label2id = from_json(f"{self.dataset_folder}/label2id.json")
        self.id2label = {v: k for k, v in self.label2id.items()}
        model = CamembertT2ForSequenceClassification.from_pretrained_body(
            prefix + model,
            time_dim=time_dim,
            nhead=nhead,
            num_layers=num_layers,
            n_cr=n_cr,
            problem_type="multi_label_classification",
            id2label=self.id2label,
            label2id=self.label2id,
        )
        # if config["freeze_body"]:
        #     for _, param in model.base_model.named_parameters():
        #         param.requires_grad=False
        # log.info(f"-------- Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        # log.info(len(config["tokenizer"]), model.config.vocab_size)
        # log.info(model.base_model.embeddings.word_embeddings)
        # if len(config["tokenizer"]) != model.config.vocab_size:
        #     log.info(f"-------- RESIZING TOKEN EMBEDDINGS \n {model.config.vocab_size} --> {len(config['tokenizer'])}")
        #     model.base_model.resize_token_embeddings(len(config["tokenizer"]))
        return model

    def collate_fn(self, data):  # data: List[Dict[str, List[float]]]
        d = self.data_collator(data)
        # log.info("ipp_id: ", d["ipp_id"])
        # log.info("dt: ", d["dt"])
        # log.info("labels: ", d["labels"])
        d["dt"] = d["dt"].reshape(-1, 1)
        d["ipp_id"] = d["ipp_id"].reshape(-1, 1)
        d["labels"] = d["labels"][self.n_cr :]
        # log.info({k:d[k].shape for k in d.keys()})
        return d

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)
        batch_sampler = (
            T2BatchSampler(eval_sampler, self._train_batch_size, self.args.dataloader_drop_last, self.n_cr)
            if hasattr(self, "n_cr")
            else None
        )
        return (
            DataLoader(
                eval_dataset,
                sampler=eval_sampler,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.collate_fn,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            if batch_sampler is None
            else DataLoader(
                eval_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        test_sampler = self._get_eval_sampler(test_dataset)
        batch_sampler = (
            T2BatchSampler(test_sampler, self._train_batch_size, self.args.dataloader_drop_last, self.n_cr)
            if hasattr(self, "n_cr")
            else None
        )
        # We use the same batch_size as for eval.
        return (
            DataLoader(
                test_dataset,
                sampler=test_sampler,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.collate_fn,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            if batch_sampler is None
            else DataLoader(
                test_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        with self.train_dataset.with_format("torch") as train_dataset:
            last_indices = (
                (train_dataset["ipp_id"] != train_dataset["ipp_id"].roll(-1)).nonzero(as_tuple=True)[0].tolist()
            )
        previous_elem = 0
        previous_schedule = -1
        for elem in last_indices:
            x = elem - previous_elem
            previous_elem = elem
            previous_schedule += x // self.args.per_device_train_batch_size
            if x % self.args.per_device_train_batch_size != 0:
                previous_schedule += 1
            self.backpropagation_schedule.append(previous_schedule)
        log.info(f"backprop schedule length: {len(self.backpropagation_schedule)}")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()
        batch_sampler = (
            T2PatientBatchSampler(
                train_sampler, self._train_batch_size, self.args.dataloader_drop_last, self.n_cr, last_indices
            )
            if hasattr(self, "n_cr")
            else None
        )
        return (
            DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                sampler=train_sampler,
                collate_fn=self.collate_fn,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
            if batch_sampler is None
            else DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
        )

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not hasattr(self.model.config, "time_dim"):
            return super()._get_train_sampler()
        else:
            if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
            ):
                return None
            elif is_torch_tpu_available():
                return get_tpu_sampler(self.train_dataset)
            else:
                return SequentialSampler(self.train_dataset)

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if not hasattr(self.model.config, "time_dim"):
            return super()._get_train_sampler()
        if self.args.use_legacy_prediction_loop:
            if self.args.local_rank != -1:
                return SequentialDistributedSampler(eval_dataset)
            else:
                return SequentialSampler(eval_dataset)
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )

    def get_labels_weight(self):
        log.info(f"using {self.label_weights_type} loss weighting")
        print(self.train_dataset["labels"])
        labels_support = np.sum(self.train_dataset["labels"], axis=0)
        maximum_support = np.max(labels_support)
        labels_support[
            labels_support <= 0
        ] = maximum_support  # TODO try with n_samples / labels_support to weigh more aggressively
        if self.label_weights_type == "max_support":
            return (
                torch.tensor(np.array([maximum_support] * len(labels_support)) / labels_support, device="cuda"),
                labels_support,
            )
        elif self.label_weights_type == "d":
            return (
                torch.tensor(
                    np.array([float(len(self.train_dataset))] * len(labels_support)) / labels_support, device="cuda"
                ),
                labels_support,
            )
        else:
            return torch.tensor([1.0] * len(labels_support), device="cuda"), labels_support

    def compute_loss(self, model, inputs, return_outputs=False):  # TODO re-enable for multilabel
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.BCEWithLogitsLoss(weight=self.labels_weight)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.float().view(-1, self.model.config.num_labels)
        )
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def flatten_report(report: Dict[str, Union[Dict[str, float], float]]) -> Dict[str, float]:
        report_2 = {}
        for k, v in report.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    report_2["_".join([k.replace(" ", "_"), k2])] = v2
            else:
                report_2[k] = v
        return report_2

    def export_predictions_samples(self, predictions: np.ndarray, label_ids: np.ndarray):
        pass

    def print_table(self, report: Dict[str, float]):
        headers = ["recall", "precision", "f1-score", "aucpr", "auroc", "brier_score", "support"]
        tablefmt = "mixed_grid"
        labels = list(self.label2id.keys()) + ["macro_avg"]
        if self.problem_type != "single_label_classification":
            labels = labels + ["micro_avg"]
        try:
            table = [[label] + [report[f"{label}_{metric}"] for metric in headers] for label in labels]
            log.info(tabulate(table, headers, tablefmt=tablefmt))
            return tabulate(table, headers, tablefmt=tablefmt)
        except Exception:
            table = [[label] + [report[f"test_{label}_{metric}"] for metric in headers] for label in labels]
            return tabulate(table, headers, tablefmt=tablefmt)

    def compute_metrics(self, eval_preds: EvalPrediction, threshold=0.5) -> Dict[str, float]:
        """Function to compute metrics at evaluation

        Args:
            eval_preds (EvalPrediction): the eval predictions and ground truth

        Returns:
            Dict[str, float]: the dictionary of metrics and their value
        """
        if not self.is_in_train:  # create plot figure
            fig_aucpr, ax_aucpr = plt.subplots(figsize=(7, 8))
            fig_auroc, ax_auroc = plt.subplots(figsize=(7, 8))
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            recall_grid = np.linspace(0.0, 1.0, 1000)
            mean_tpr = np.zeros_like(fpr_grid)
            mean_precision = np.zeros_like(recall_grid)
            mean_thresholds = np.zeros_like(fpr_grid)
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (lines,) = ax_aucpr.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.3)
                ax_aucpr.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
        activation = (
            torch.nn.Sigmoid() if self.problem_type == "multi_label_classification" else torch.nn.Softmax(dim=1)
        )
        predictions, label_ids = eval_preds
        probs = activation(torch.Tensor(predictions))
        if self.problem_type == "multi_label_classification":
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= threshold)] = 1.0
        else:
            y_pred = np.argmax(probs, axis=1)
        flat_report = self.flatten_report(
            classification_report(label_ids, y_pred, output_dict=True, target_names=self.label2id.keys())
        )
        ps = average_precision_score(label_ids, probs, average=None)
        roc_auc = roc_auc_score(label_ids, probs, average=None)
        flat_report["micro_avg_brier_score"] = brier_score_loss(label_ids.ravel(), probs.ravel())
        brier_scores = [brier_score_loss(label_ids[:, i], probs[:, i]) for i in self.id2label.keys()]
        flat_report["macro_avg_brier_score"] = np.mean(brier_scores)
        flat_report["macro_avg_auroc"] = np.mean(roc_auc)
        flat_report["macro_avg_aucpr"] = np.mean(ps)
        flat_report["best_metric"] = (
            max(flat_report["macro_avg_aucpr"], self.state.best_metric)
            if self.state.best_metric is not None
            else flat_report["macro_avg_aucpr"]
        )
        # flat_report["weighted_avg_aucpr"] = np.sum([ps[i]*self.eval_supports[i] for i in range(len(ps))])/np.sum(self.eval_supports)
        precisions_micro, recalls_micro, thresholds_aucpr_micro = precision_recall_curve(
            label_ids.ravel(), probs.ravel()
        )
        fpr_micro, tpr_micro, thresholds_micro = roc_curve(label_ids.ravel(), probs.ravel())
        flat_report["micro_avg_auroc"] = auc(fpr_micro, tpr_micro)
        flat_report["micro_avg_aucpr"] = auc(recalls_micro, precisions_micro)
        c1 = self.state.best_metric is None
        c2 = True if c1 else flat_report[self.args.metric_for_best_model] > self.state.best_metric
        if not self.is_in_train:  # plot micro-average aucpr/auroc
            display_aucpr = PrecisionRecallDisplay(
                precision=precisions_micro, recall=recalls_micro, average_precision=flat_report["micro_avg_aucpr"]
            )
            display_aucpr.plot(
                ax=ax_aucpr, name="Micro-average precision-recall", color="gold", linestyle=":", linewidth=4
            )
            display_auroc = RocCurveDisplay(fpr=fpr_micro, tpr=tpr_micro, roc_auc=flat_report["micro_avg_auroc"])
            display_auroc.plot(ax=ax_auroc, name="Micro-average ROC curve", color="gold", linestyle=":", linewidth=4)
            ax_auroc.plot(fpr_micro, thresholds_micro, linestyle='dashed', color="gold", alpha=0.6)
        for (i, label), color in zip(self.id2label.items(), COLORS):
            flat_report[f"{label}_brier_score"] = brier_scores[i]
            flat_report[f"{label}_auroc"] = roc_auc[i]
            flat_report[f"{label}_aucpr"] = ps[i]
            precisions, recalls, thresholds = precision_recall_curve(label_ids[:, i], probs[:, i])
            fpr, tpr, thresholds_roc = roc_curve(label_ids[:, i], probs[:, i])
            if (c1 or c2) and self.is_in_train:
                np.savez(
                    f"{self.args.output_dir}/aucpr_{i}.npz",
                    recalls=recalls,
                    precisions=precisions,
                    thresholds=thresholds,
                    aucpr=ps[i],
                    label=label,
                )
                np.savez(
                    f"{self.args.output_dir}/auroc_{i}.npz",
                    fpr=fpr,
                    tpr=tpr,
                    thresholds=thresholds_roc,
                    auroc=roc_auc[i],
                    label=label,
                )
            if not self.is_in_train:  # plot label-wise aucpr/auroc and thresholds
                display_aucpr = PrecisionRecallDisplay(
                    recall=recalls,
                    precision=precisions,
                    average_precision=ps[i],
                )
                display_aucpr.plot(ax=ax_aucpr, name=f"Precision-recall for class {label}", color=color)
                display_auroc = RocCurveDisplay(
                    fpr=fpr,
                    tpr=tpr,
                    roc_auc=roc_auc[i],
                )
                display_auroc.plot(ax=ax_auroc, name=f"ROC curve for class {label}", color=color)
                ax_auroc.plot(fpr, thresholds_roc, linestyle='dashed', color=color, alpha=0.6)
                mean_tpr += np.interp(fpr_grid, fpr, tpr)
                mean_thresholds += np.interp(fpr_grid, fpr, thresholds_roc)
                mean_precision += np.interp(recall_grid, recalls[::-1], precisions[::-1])
        if not self.is_in_train:  # plot macro-average aucpr/auroc
            mean_tpr /= len(self.id2label)
            mean_thresholds /= len(self.id2label)
            mean_precision /= len(self.id2label)
            display_auroc = RocCurveDisplay(fpr=fpr_grid, tpr=mean_tpr, roc_auc=auc(fpr_grid, mean_tpr))
            display_aucpr = PrecisionRecallDisplay(
                recall=recall_grid, precision=mean_precision, average_precision=auc(recall_grid, mean_precision)
            )
            display_aucpr.plot(
                ax=ax_aucpr, name="Macro-average precision-recall", color="silver", linestyle=":", linewidth=4
            )
            display_auroc.plot(ax=ax_auroc, name="Macro-average ROC curve", color="silver", linestyle=":", linewidth=4)
            ax_auroc.plot(fpr_grid, mean_thresholds, linestyle="dashed", color="silver", alpha=0.6)
            handles_aucpr, labels_aucpr = display_aucpr.ax_.get_legend_handles_labels()
            handles_auroc, labels_auroc = display_auroc.ax_.get_legend_handles_labels()
            handles_aucpr.extend([lines])
            labels_aucpr.extend(["iso-f1 curves"])
            # set the legend and the axes
            ax_aucpr.set_xlim([0.0, 1.0])
            ax_aucpr.set_ylim([0.0, 1.05])
            ax_aucpr.legend(handles=handles_aucpr, labels=labels_aucpr, loc="best")
            ax_aucpr.set_title("Multi-label precision-recall curves")
            ax_aucpr.grid(True)
            ax_auroc.set_xlim([0.0, 1.0])
            ax_auroc.set_ylim([0.0, 1.05])
            ax_auroc.legend(handles=handles_auroc, labels=labels_auroc, loc="best")
            ax_auroc.set_title("Multi-label ROC curves")
            ax_auroc.grid(True)
            fig_aucpr.savefig(f"{self.args.output_dir}/figures/aucpr_test")
            fig_auroc.savefig(f"{self.args.output_dir}/figures/auroc_test")
        if (c1 or c2) and self.is_in_train:  # best or first model: save aucpr and auroc
            np.savez(
                f"{self.args.output_dir}/aucpr_micro.npz",
                recalls=recalls_micro,
                precisions=precisions_micro,
                thresholds=thresholds_aucpr_micro,
                ps=flat_report["micro_avg_aucpr"],
            )
            np.savez(
                f"{self.args.output_dir}/auroc_micro.npz",
                fpr=fpr_micro,
                tpr=tpr_micro,
                thresholds=thresholds_micro,
                auroc=flat_report["micro_avg_auroc"],
            )
        table = self.print_table(flat_report)
        if self.is_in_train:
            with open(f"{self.args.output_dir}/eval_metrics.txt", "a") as f:
                f.write(f"epoch {str(self.state.epoch)}:\n")
                f.write(table)
                f.write("\n\n")
        return flat_report



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    gc.collect()
    torch.cuda.empty_cache()
    config = {
        "model" : "/home/bourhani@clb.loc/saepred/models/output/optuna/run-31/checkpoint-127136",
        "dataset_folder" : "/home/bourhani@clb.loc/saepred/data_dounya/featurized/data_brut",
        "n_cr" : 7,
        "nhead" : 4,
        "time_dim" : 8,
        "num_layers" : 10,
        "padding" : "longest",
        "label_weights_type": "aggressive",
        "resume": False,
        "args" : {  
            "output_dir" : "/home/bourhani@clb.loc/saepred/models/output/benchmark_dounya/OncoBERT_initial",
            "per_device_train_batch_size": 16,
            "learning_rate": 9.678963007161723e-06,
            "num_train_epochs": 10,
            "seed": 42,
            "fp16": True,
            "evaluation_strategy": "epoch",
            "logging_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_avg_aucpr",
            "per_device_eval_batch_size": 32
            }
    }

    trainer = SaeTrainer.from_config(config)
    # trainer = SaeTrainer.from_config(cfg)
    # print(trainer)
    # if trainer.freeze_body:
    # for _, param in trainer.model.base_model.named_parameters():
    #     param.requires_grad=False

    trainer.train(resume_from_checkpoint=config["resume"])
    trainer.save_state()
    trainer.save_model()

    test_dataset = T2Dataset(
        load_from_disk(f"{trainer.dataset_folder}/test").rename_column("dt_since_first", "dt")._data, n_cr=trainer.n_cr
    )
    test_predictions = trainer.predict(test_dataset)
    with open(f"{trainer.args.output_dir}/test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(trainer.print_table(test_predictions.metrics))
    return

if __name__ == "__main__":
    main()