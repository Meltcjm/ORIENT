# Portions adapted from LLaVA/FastChat/Stanford Alpaca, Apache-2.0.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li.
# See licenses/Apache-2.0.txt and THIRD_PARTY_NOTICES.md.
"""LLaVA-1.5 LoRA setup; model and optimizer persist across all stages."""
from copy import copy
from pathlib import Path

import torch
from torch.utils.data import Dataset


def parse_native_arguments(args, extra):
    from transformers import HfArgumentParser
    from llava.train.train import ModelArguments, DataArguments, TrainingArguments
    defaults = [
        "--model_name_or_path", str(args.model_name_or_path), "--version", "v1",
        "--data_path", str(args.data_path), "--image_folder", str(args.image_folder),
        "--vision_tower", str(args.vision_tower), "--mm_vision_select_layer", "-2",
        "--mm_projector_type", "mlp2x_gelu", "--mm_use_im_patch_token", "False",
        "--output_dir", str((args.out / "training").resolve()), "--num_train_epochs", "1",
        "--per_device_train_batch_size", "8", "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-4", "--bf16", "True", "--tf32", "True",
        "--gradient_checkpointing", "True", "--lazy_preprocess", "True",
        "--dataloader_num_workers", "4", "--group_by_modality_length", "True",
        "--lora_enable", "True", "--lora_r", "128", "--lora_alpha", "256",
        "--model_max_length", "2048", "--image_aspect_ratio", "pad",
        "--seed", str(args.seed), "--remove_unused_columns", "False", "--report_to", "none",
        "--lr_scheduler_type", "cosine", "--warmup_ratio", ".03", "--weight_decay", "0",
        "--save_strategy", "no",
    ]
    model_args, data_args, training_args = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    ).parse_args_into_dataclasses(args=defaults + list(extra))
    if training_args.split_batches:
        raise ValueError("split_batches changes the effective global batch; keep it disabled")
    if training_args.dispatch_batches and training_args.world_size > 1:
        raise ValueError("dispatch_batches cannot concatenate native multimodal batches with different lengths")
    if training_args.label_smoothing_factor != 0:
        raise ValueError("label_smoothing_factor must be zero; native multimodal labels expand inside the model")
    if training_args.n_gpu > 1:
        raise ValueError("use torchrun for multiple GPUs, or expose one GPU via CUDA_VISIBLE_DEVICES")
    if (training_args.num_train_epochs != 1 or training_args.max_steps > 0
            or training_args.auto_find_batch_size or training_args.include_tokens_per_second
            or training_args.dataloader_drop_last):
        raise ValueError("ORIENT needs one complete epoch and a fixed global batch")
    if (training_args.bits != 16 or not training_args.lora_enable
            or "mpt" in model_args.model_name_or_path.lower() or not model_args.vision_tower):
        raise ValueError("this backend supports LLaVA-1.5 Llama with 16-bit LoRA")
    if model_args.tune_mm_mlp_adapter or training_args.freeze_mm_mlp_adapter:
        raise ValueError("ORIENT uses the native LoRA recipe with a trainable multimodal projector")
    if training_args.remove_unused_columns:
        raise ValueError("native multimodal columns must be retained")
    if (model_args.version != "v1" or model_args.mm_use_im_start_end
            or model_args.mm_use_im_patch_token or training_args.fsdp
            or training_args.lora_weight_path or training_args.resume_from_checkpoint
            or training_args.do_eval or training_args.dataloader_persistent_workers
            or str(training_args.save_strategy) not in ("no", "IntervalStrategy.NO")):
        raise ValueError("this release supports fresh v1 LoRA training without intermediate checkpoints")
    if (model_args.mm_vision_select_layer != -2 or model_args.mm_vision_select_feature != "patch"
            or model_args.mm_projector_type != "mlp2x_gelu" or model_args.mm_patch_merge_type != "flat"
            or model_args.pretrain_mm_mlp_adapter or data_args.image_aspect_ratio != "pad"):
        raise ValueError("use a full LLaVA-1.5 checkpoint with the fixed paired-scoring preprocessing")
    if (Path(training_args.output_dir).resolve() != (args.out / "training").resolve()
            or Path(data_args.data_path).resolve() != args.data_path.resolve()
            or Path(data_args.image_folder).resolve() != args.image_folder.resolve()
            or training_args.seed != args.seed):
        raise ValueError("native arguments must not override bound data, output or seed")
    if (model_args.model_name_or_path != str(args.model_name_or_path)
            or model_args.vision_tower != str(args.vision_tower)):
        raise ValueError("use the ORIENT model/vision arguments to set checkpoint identities")
    plugin = getattr(training_args, "deepspeed_plugin", None)
    if plugin is not None:
        zero = plugin.deepspeed_config.setdefault("zero_optimization", {})
        if int(zero.get("stage", 0)) > 2:
            raise ValueError("online distributed scoring supports replicated parameters (ZeRO <= 2)")
        zero["overlap_comm"] = False
    return model_args, data_args, training_args


def build_model(model_args, data_args, training_args, attn_implementation):
    import transformers
    from peft import LoraConfig, get_peft_model
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava import conversation as conversation_lib
    from llava.train.train import find_all_linear_names

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    config = LlavaConfig.from_pretrained(model_args.model_name_or_path)
    if not getattr(config, "mm_vision_tower", None):
        raise ValueError("initial model must be a full multimodal LLaVA checkpoint")
    if (getattr(config, "mm_projector_type", "linear") != "mlp2x_gelu"
            or getattr(config, "mm_patch_merge_type", "flat") != "flat"):
        raise ValueError("checkpoint must use the LLaVA-1.5 mlp2x_gelu projector and flat patch merge")
    config.mm_vision_tower = model_args.vision_tower
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        config=config, attn_implementation=attn_implementation, torch_dtype=dtype)
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def require_grad(_module, _inputs, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(require_grad)
    model = get_peft_model(model, LoraConfig(
        r=training_args.lora_r, lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model), lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias, task_type="CAUSAL_LM"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision = model.get_vision_tower()
    vision.to(dtype=dtype, device=training_args.device)
    data_args.image_processor = vision.image_processor
    data_args.is_multimodal = True
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = False
    model.config.freeze_mm_mlp_adapter = False
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    return model, tokenizer


def save_native_model(trainer):
    model, args = trainer.model, trainer.args
    model.config.use_cache = True
    if args.process_index == 0:
        # Supported ZeRO <= 2 keeps parameters replicated; PEFT handles bias selection.
        state = {key: value.detach().cpu().clone() for key, value in model.named_parameters()
                 if value.requires_grad or "lora_" in key}
        non_lora = {key: value for key, value in state.items() if "lora_" not in key}
        model.config.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir, state_dict=state)
        trainer.tokenizer.save_pretrained(args.output_dir)
        torch.save(non_lora, Path(args.output_dir) / "non_lora_trainables.bin")


def training_configuration(args):
    # Keep reproducibility settings, without raw CLI arguments or credential fields.
    fields = (
        "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate",
        "num_train_epochs", "max_steps", "lora_r", "lora_alpha", "lora_dropout", "lora_bias",
        "warmup_ratio", "warmup_steps", "lr_scheduler_type", "lr_scheduler_kwargs",
        "weight_decay", "optim", "optim_args", "adam_beta1", "adam_beta2", "adam_epsilon",
        "max_grad_norm", "seed", "data_seed", "model_max_length", "mm_projector_lr",
        "bf16", "fp16", "tf32", "gradient_checkpointing", "gradient_checkpointing_kwargs",
        "group_by_modality_length", "dataloader_num_workers", "dataloader_pin_memory",
        "ddp_find_unused_parameters", "ddp_bucket_cap_mb", "ddp_broadcast_buffers",
        "full_determinism", "torch_compile", "torch_compile_backend", "torch_compile_mode")
    values = args.to_dict()
    result = {key: values[key] for key in fields if key in values}
    plugin = getattr(args, "deepspeed_plugin", None)
    result["deepspeed"] = plugin.deepspeed_config if plugin is not None else None
    return result


class OnlineSelectedDataset(Dataset):
    def __init__(self, source_dataset, stages):
        self.source_dataset, self.stages = source_dataset, stages
        self.assignment = [-1] * sum(stage.K_t for stage in stages)

    def __len__(self):
        return len(self.assignment)

    def __getitem__(self, slot):
        index = self.assignment[int(slot)]
        if index < 0:
            raise RuntimeError("training sample requested before selection")
        return self.source_dataset[index]

    def assign(self, stage, selected):
        if (len(selected) != stage.K_t or len(set(selected)) != stage.K_t
                or not set(selected) <= set(stage.P_t)):
            raise ValueError("selected rows must be unique members of this stage")
        if any(i >= 0 for i in self.assignment[stage.offset:stage.offset + stage.K_t]):
            raise RuntimeError("stage was already assigned")
        self.assignment[stage.offset:stage.offset + stage.K_t] = selected

    def stage_view(self, stage):
        indices = self.assignment[stage.offset:stage.offset + stage.K_t]
        if any(i < 0 for i in indices):
            raise RuntimeError("stage loader requested before selection")
        view = copy(self.source_dataset)
        view.list_data_dict = [self.source_dataset.list_data_dict[i] for i in indices]
        return view


class OnlineStageDataLoader:
    def __init__(self, trainer):
        self.trainer, self.dataset = trainer, trainer.train_dataset
        self.batch_size = trainer._train_batch_size

    def __len__(self):
        return len(self.dataset) // (self.batch_size * int(self.trainer.args.world_size))

    def __iter__(self):
        for stage in self.dataset.stages:
            controller = self.trainer.orient_controller
            if controller.active is None or controller.active["stage"] is not stage:
                raise RuntimeError("native loader advanced before stage selection")
            loader = self.trainer._stage_dataloader(self.dataset.stage_view(stage))
            try:
                yield from loader
            finally:
                self.trainer.accelerator._dataloaders.remove(loader)


def make_trainer_class():
    from llava.train.llava_trainer import LLaVATrainer

    class OrientTrainer(LLaVATrainer):
        def get_train_dataloader(self):
            return OnlineStageDataLoader(self)

        def _stage_dataloader(self, dataset):
            original = self.train_dataset
            try:
                self.train_dataset = dataset
                return super().get_train_dataloader()
            finally:
                self.train_dataset = original

    return OrientTrainer


def make_stage_callback(controller):
    from transformers import TrainerCallback

    class StageCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            controller.record_training_configuration(controller.trainer)
            controller.begin_stage(controller.stages[0], controller.trainer)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            active = controller.active
            if active is None:
                return control
            stage = active["stage"]
            target = active["global_step_before"] + stage.updates
            if state.global_step < target:
                return control
            if state.global_step != target:
                raise RuntimeError("native optimizer crossed a stage boundary")
            controller.end_stage(stage, controller.trainer)
            if stage.t + 1 < len(controller.stages):
                controller.begin_stage(controller.stages[stage.t + 1], controller.trainer)
            else:
                controller.finish(controller.trainer)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if controller.active is not None or not controller.finished:
                raise RuntimeError("training ended before completing all stages")
            return control

    return StageCallback()
