import os
import torch
import torch.nn as nn
import datetime
from functools import partial
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.utils.data import Dataset, Sampler, DataLoader

from transformers import Trainer
from trl.trainer import SFTTrainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, logger, is_accelerate_available, is_datasets_available
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from transformers.trainer_pt_utils import AcceleratorConfig
from typing import List, Optional
from datetime import timedelta

# GKD imports
from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.trainer import DPOTrainer
import random
from transformers import Trainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs

if is_datasets_available():
    import datasets

from llava.utils import rank0_print


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")

        # create accelerator object
        #self.accelerator = Accelerator(
        #    dispatch_batches=self.args.dispatch_batches, split_batches=self.args.split_batches, deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, kwargs_handlers=[accelerator_kwargs]
        #)

        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=self.args.accelerator_config.dispatch_batches,
            split_batches=self.args.accelerator_config.split_batches
        )
        self.accelerator = Accelerator(
           dataloader_config=dataloader_config, 
           deepspeed_plugin=self.args.deepspeed_plugin, 
           gradient_accumulation_plugin=gradient_accumulation_plugin, 
           kwargs_handlers=[accelerator_kwargs]
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # maybe shuffle here

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }


        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            rank = int(os.environ.get("RANK", 0))
            dataloader_params["worker_init_fn"] = partial( # this torch version does not pass these added two arguments transformer needs
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=rank
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


class LLaVADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            # super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)
            # print(type(model))
            # from transformers.modeling_utils import unwrap_model
            # print(type(unwrap_model(model)))
            # print(unwrap_model(model).config)
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)


class LLaVAGKDTrainer(GKDTrainer):
    """
    GKD Trainer adapted for multimodal LLaVA models.
    
    Overrides compute_loss and training_step to properly handle multimodal inputs
    (images, image_sizes, modalities) that LLaVA models require.
    """
    
    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        # Use provided dataset or fall back to self.train_dataset
        dataset = dataset if dataset is not None else self.train_dataset

        # return standard deterministic sampler with no shuffling
        # create a sampler that returns the indices in order
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        return sampler

        if dataset is None or not has_length(dataset):
            return None
        
        if getattr(self.args, "group_by_modality_length", False):
            lengths = dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler(dataset)

    def _get_multimodal_inputs(self, inputs):
        """
        Extract multimodal-specific inputs from the batch without removing them.
        Returns a dict with multimodal inputs.
        """
        mm_inputs = {}
        
        # Extract image-related inputs
        for key in ["images", "image_sizes", "modalities", "image_grids", 
                    "pixel_attention_mask", "spatial_shapes"]:
            if key in inputs:
                mm_inputs[key] = inputs[key]
                
        return mm_inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute GKD loss with multimodal support.
        
        This overrides the parent compute_loss to pass images and other multimodal
        inputs to both student and teacher models. Mirrors the original GKDTrainer
        compute_loss but with multimodal handling.
        """
        from trl.trainer.utils import empty_cache
        
        # Get multimodal inputs
        mm_inputs = self._get_multimodal_inputs(inputs)
        modalities = mm_inputs["modalities"]
    
        if self.use_liger_gkd_loss:
            # Liger fused kernel path - forward only through the base models (avoid lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_student, "get_decoder") and unwrapped_student.get_decoder() is not None:
                base_student = unwrapped_student.get_decoder()
            else:
                base_student = getattr(
                    unwrapped_student, getattr(unwrapped_student, "base_model_prefix", "model"), unwrapped_student
                )

            # Build forward kwargs for base model - note: base models may not handle multimodal
            # For LLaVA, the full model handles image injection, so we use the full model forward
            forward_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "use_cache": False,
            }
            forward_kwargs.update(mm_inputs)
            
            # For multimodal models, we need to use the full model, not base
            student_outputs = model(**forward_kwargs)

            self.teacher_model.eval()
            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            unwrapped_teacher.eval()
            
            with torch.no_grad():
                teacher_outputs = unwrapped_teacher(**forward_kwargs)

            # hidden states (shifted)
            student_hidden = student_outputs.hidden_states[-1][:, :-1] if hasattr(student_outputs, 'hidden_states') and student_outputs.hidden_states else student_outputs.logits[:, :-1, :]
            teacher_hidden = teacher_outputs.hidden_states[-1][:, :-1] if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states else teacher_outputs.logits[:, :-1, :]

            # labels mask and labels (shifted)
            labels_mask = inputs["labels"] != -100
            masked_input_ids = torch.where(
                labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
            )
            true_labels = masked_input_ids[:, 1:].contiguous()

            # heads
            student_head = unwrapped_student.get_output_embeddings()
            teacher_head = unwrapped_teacher.get_output_embeddings()

            # liger fused jsd loss
            loss = self.liger_jsd_loss(
                student_input=student_hidden,
                student_weight=student_head.weight,
                teacher_input=teacher_hidden,
                teacher_weight=teacher_head.weight,
                true_labels=true_labels,
                student_bias=getattr(student_head, "bias", None),
                teacher_bias=getattr(teacher_head, "bias", None),
            )

            del student_hidden, teacher_hidden, true_labels
        else:
            # Unwrap student model to ensure multimodal kwargs are passed through
            
            # compute student output
            # import pdb; pdb.set_trace()
            x = inputs["input_ids"][0]
            print(self.tokenizer.decode(x[x>0]))
            if "images" not in mm_inputs:
                import pdb; pdb.set_trace()
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=mm_inputs["images"],
                image_sizes=mm_inputs["image_sizes"],
                modalities=mm_inputs["modalities"],
                return_token_indices=True if "image" in modalities else False,
            )
            
            # Compute teacher output in eval mode with multimodal inputs
            # Unwrap teacher model to avoid accelerate wrapper issues with multimodal kwargs
            # Multimodal: logits have expanded vision tokens, labels have original length.
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    images=mm_inputs["images"],
                    image_sizes=mm_inputs["image_sizes"],
                    modalities=mm_inputs["modalities"],
                )

            # slice the logits for the generated tokens using the inputs["prompts"] lengths
            prompt_lengths = inputs["prompts"].shape[1]

            if "image" in modalities:    
                student_outputs, token_indices = student_outputs
                B = student_outputs.logits.shape[0]
                device = student_outputs.logits.device

                # Project expanded logits → text space
                student_text_logits = []
                teacher_text_logits = []

                for b in range(B):
                    text_positions = torch.stack(token_indices[b]["text"]).to(device)
                    student_text_logits.append(student_outputs.logits[b, text_positions])
                    teacher_text_logits.append(teacher_outputs.logits[b, text_positions])

                student_text_logits = torch.stack(student_text_logits)   # (B, L_text, V)
                teacher_text_logits = torch.stack(teacher_text_logits)   # (B, L_text, V)
                shifted_labels = inputs["labels"][:, prompt_lengths + 1:]
            else:
                student_text_logits = student_outputs.logits
                teacher_text_logits = teacher_outputs.logits
                shifted_labels = inputs["labels"][:, prompt_lengths ]

            #start = P - 1   # because logits already shifted once
            shifted_student_logits = student_text_logits[:, prompt_lengths - 1 : -1, :]
            shifted_teacher_logits = teacher_text_logits[:, prompt_lengths - 1 : -1, :]
            


            # Compute generalized JSD loss (shapes now match: [batch, L, vocab] and [batch, L])
            loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
            )

        # Empty cache as in original
        empty_cache()

        return (loss, student_outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: dict, num_items_in_batch: int = None
    ) -> torch.Tensor:
        """
        Perform a training step with multimodal support.
        
        This mirrors the original GKDTrainer.training_step but adds multimodal
        input handling for generation and forward passes.
        """
    
        # Get multimodal inputs for generation
        mm_inputs = self._get_multimodal_inputs(inputs)

        # Sequence-level KD: generate from teacher
        if self.seq_kd:
            with unwrap_model_for_generation(
                self.teacher_model,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self._generate_on_policy_outputs_multimodal(
                    unwrapped_model, inputs, mm_inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        # On-policy: with probability lmbda, generate from student
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(
                model,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self._generate_on_policy_outputs_multimodal(
                    unwrapped_model, inputs, mm_inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        # Call Trainer.training_step directly, bypassing GKDTrainer.training_step
        # which would do on-policy generation again (without multimodal support)
        loss = SFTTrainer.training_step(self, model, inputs, num_items_in_batch)
        # loss = Trainer.training_step(self, model, inputs, num_items_in_batch)
        
        return loss

    def _generate_on_policy_outputs_multimodal(self, model, inputs, mm_inputs, generation_config, pad_token_id=None):
        """
        Generate on-policy outputs with multimodal conditioning.
        
        Mirrors the original GKDTrainer.generate_on_policy_outputs but passes
        multimodal inputs (images, etc.) to the generate call.
        """
        
        print("Generating on-policy outputs with multimodal conditioning")
        # Get prompt inputs
        # For LLaVA, if "prompts" not available, we need to extract prompt from input_ids
        # using labels (where labels == -100 indicates prompt tokens)
        
        prompts = inputs["prompts"]
        prompts_attention_mask = inputs.get("prompts_attention_mask", None)


        # Build generate kwargs
        generate_kwargs = {
            "generation_config": generation_config,
            "return_dict_in_generate": True,
        }
        
        if prompts_attention_mask is not None:
            generate_kwargs["attention_mask"] = prompts_attention_mask
        
        # Add multimodal inputs for conditioned generation
        # generate_kwargs.update(mm_inputs)
        
        # Generate output
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        generated_outputs = model.generate(
            inputs=inputs["prompts"],
            images=mm_inputs["images"],
            image_sizes=mm_inputs["image_sizes"],
            modalities=mm_inputs["modalities"],
            eos_token_id=107,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
            use_cache=True
            # **generate_kwargs
        )

        import pdb; pdb.set_trace()
        
        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences

        # decode sentence
        p, s = inputs["prompts"][0], generated_outputs.sequences[0]
        self.tokenizer.decode(p[p>0])
        self.tokenizer.decode(s[s>0])
        
        
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        # and set labels to -100 for padding
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            if getattr(self.args, "lora_enable", False):
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVAGKDTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVAGKDTrainer, self)._save(output_dir, state_dict)
