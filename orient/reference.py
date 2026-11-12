"""Cache frozen-reference losses with the same paired scoring used online."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from .cache import load_legacy_reference, scoring_contract, write_reference_cache
from .data import load_rows
from .scoring import score_rows
from .reference_shards import bind_shards, ordered_scores, read_completed, save_chunk


def load_model(base_path, reference_path, vision_tower, model_max_length, dtype, attn_implementation):
    from transformers import AutoTokenizer
    from peft import PeftModel
    from llava import conversation as conversation_lib
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

    reference = Path(reference_path)
    adapter = (reference / "adapter_config.json").is_file()
    config = LlavaConfig.from_pretrained(reference)
    if (getattr(config, "mm_projector_type", "linear") != "mlp2x_gelu"
            or getattr(config, "mm_patch_merge_type", "flat") != "flat"):
        raise ValueError("checkpoint must use the LLaVA-1.5 mlp2x_gelu projector and flat patch merge")
    config.mm_vision_tower = str(vision_tower)
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_path if adapter else reference, config=config,
        torch_dtype=dtype, attn_implementation=attn_implementation)
    vision = model.get_vision_tower()
    if not vision.is_loaded:
        vision.load_model()
    if adapter:
        state = torch.load(reference / "non_lora_trainables.bin", map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or any(
                not isinstance(k, str) or not torch.is_tensor(v) or not bool(torch.isfinite(v).all())
                for k, v in state.items()):
            raise ValueError("reference non-LoRA state must contain finite tensors")
        state = {k[11:] if k.startswith("base_model.") else k: v for k, v in state.items()}
        state = {k[6:] if k.startswith("model.model.") else k: v for k, v in state.items()}
        projector_keys = {"model.mm_projector." + key for key in model.get_model().mm_projector.state_dict()}
        if not projector_keys.issubset(state):
            raise ValueError("reference adapter is missing multimodal projector weights")
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError("reference contains unexpected non-LoRA weights")
        model = PeftModel.from_pretrained(model, reference)
    tokenizer = AutoTokenizer.from_pretrained(base_path, model_max_length=model_max_length,
                                               padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    model.config.use_cache = False
    model.config.image_aspect_ratio = "pad"
    model.config.tokenizer_model_max_length = model_max_length
    model.config.tokenizer_padding_side = "right"
    model.config.mm_use_im_start_end = False
    model.requires_grad_(False).eval().to(device="cuda", dtype=dtype)
    model.get_vision_tower().to(device="cuda", dtype=dtype)
    return model, tokenizer


def launch_shards(gpus, argv):
    """Use independent interpreters; a killed worker cannot poison a process pool."""
    worker_args = list(argv)
    i = 0
    while i < len(worker_args):
        token = worker_args[i]
        if token == "--gpus":
            del worker_args[i:i + 2]
        elif token.startswith("--gpus="):
            del worker_args[i]
        else:
            i += 1
    processes = []
    try:
        for shard, gpu in enumerate(gpus):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                environment.setdefault(key, "8")
            command = [sys.executable, "-B", "-m", "orient.reference", *worker_args,
                       "--shard", str(shard), "--num-shards", str(len(gpus))]
            processes.append(subprocess.Popen(command, env=environment))
        while True:
            statuses = [process.poll() for process in processes]
            failed = [(i, code) for i, code in enumerate(statuses) if code not in (None, 0)]
            if failed:
                raise RuntimeError(f"reference workers failed: {failed}; saved chunks can be resumed")
            if all(code == 0 for code in statuses):
                return
            time.sleep(0.2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def score_shard(args, rows, data_args, legacy_scores, done):
    shard = args.shard if args.shard is not None else 0
    assigned = range(shard, len(rows), args.num_shards)
    imported = [i for i in assigned if i not in done and legacy_scores[i] is not None]
    for start in range(0, len(imported), args.chunk_size):
        indices = imported[start:start + args.chunk_size]
        values = [legacy_scores[i] for i in indices]
        save_chunk(args.work_dir, indices, values, rows)
        done.update(zip(indices, values))
    pending = [i for i in assigned if i not in done]
    print(f"Reference shard {shard}/{args.num_shards}: {len(pending)} rows to score", flush=True)
    if not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"{len(pending)} reference rows require CUDA scoring; saved chunks can be resumed")
    model, tokenizer = load_model(args.model_name_or_path, args.reference_model, args.vision_tower,
                                  args.model_max_length, getattr(torch, args.dtype), args.attn_implementation)
    data_args.image_processor = model.get_vision_tower().image_processor
    from .acceleration import ScoringRuntime
    from .scoring_inputs import ScoringInputCache
    runtime = ScoringRuntime(answer_sliced=args.answer_sliced_logits)
    input_cache = ScoringInputCache()
    started = time.monotonic()
    for start in range(0, len(pending), args.chunk_size):
        indices = pending[start:start + args.chunk_size]
        values = score_rows(model, tokenizer, [rows[i] for i in indices], str(args.image_folder),
                            data_args, args.score_batch_size, input_cache=input_cache, runtime=runtime)
        save_chunk(args.work_dir, indices, values, rows)
        done.update(zip(indices, values))
        completed = start + len(indices)
        seconds_per_row = (time.monotonic() - started) / completed
        print(f"Reference shard {shard}/{args.num_shards}: {completed}/{len(pending)} saved, "
              f"{seconds_per_row * 1000:.0f} ms/row, "
              f"ETA {(len(pending) - completed) * seconds_per_row / 3600:.2f} h", flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("data-path", "model-name-or-path", "reference-model", "vision-tower", "image-folder", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--work-dir", type=Path, help="durable chunks; defaults to <out>.parts; reuse to resume")
    parser.add_argument("--gpus", help="comma-separated CUDA device IDs/UUIDs; independent workers then merge")
    parser.add_argument("--shard", type=int, help="manually run one shard, then use --merge")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="merge completed chunks without loading a model")
    parser.add_argument("--answer-sliced-logits", action="store_true")
    parser.add_argument("--legacy-cache", type=Path,
                        help="reuse a bound paired_tv_v1 cache; assumes original checkpoints/images are unchanged")
    parser.add_argument("--attn-implementation", choices=("eager", "flash_attention_2"), default="flash_attention_2")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("reference output already exists")
    if args.model_max_length <= 1 or min(args.score_batch_size, args.chunk_size, args.num_shards) <= 0:
        parser.error("model-max-length must exceed 1; batch size, chunk size and num-shards must be positive")
    if args.shard is not None and not 0 <= args.shard < args.num_shards:
        parser.error("shard must be in [0, num-shards)")
    if args.shard is None and args.num_shards != 1:
        parser.error("num-shards requires shard; merging does not need a shard count")
    if args.merge and args.shard is not None:
        parser.error("merge and shard are separate operations")
    if args.gpus is not None and (args.merge or args.shard is not None):
        parser.error("gpus cannot be combined with shard or merge")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("use --gpus or independent --shard processes for reference generation, not torchrun")
    args.work_dir = args.work_dir or Path(str(args.out) + ".parts")
    if args.gpus is not None:
        gpus = [device.strip() for device in args.gpus.split(",")]
        if any(not device or device.startswith("-") for device in gpus) or len(set(gpus)) != len(gpus):
            parser.error("gpus must list distinct nonempty CUDA device IDs/UUIDs")
        launch_shards(gpus, argv)
        args.merge = True
    rows = load_rows(args.data_path)
    data_args = SimpleNamespace(is_multimodal=True, image_aspect_ratio="pad", mm_use_im_start_end=False,
                                image_folder=str(args.image_folder))
    model_args = SimpleNamespace(model_name_or_path=str(args.model_name_or_path),
                                 vision_tower=str(args.vision_tower), version="v1", mm_use_im_start_end=False)
    training_args = SimpleNamespace(model_max_length=args.model_max_length,
                                    bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16")
    contract = scoring_contract(model_args, data_args, training_args, args.reference_model)
    scores, legacy_import = ([None] * len(rows), None)
    if args.legacy_cache:
        scores, legacy_import = load_legacy_reference(args.legacy_cache, rows, contract)
        print(f"Legacy reference: reusing {legacy_import['reused_rows']}/{len(rows)} rows. "
              "Generation-time checkpoint hashes are unavailable; reuse assumes unchanged checkpoints/images.",
              flush=True)
    generation = {"image_folder": str(args.image_folder.resolve()), "score_batch_size": args.score_batch_size,
                  "chunk_size": args.chunk_size, "attn_implementation": args.attn_implementation,
                  "answer_sliced_logits": args.answer_sliced_logits, "length_bucketing": True}
    bind_shards(args.work_dir, rows, contract, generation, legacy_import)
    done = read_completed(args.work_dir, rows)
    for index, value in done.items():
        if scores[index] is not None and scores[index] != value:
            raise ValueError(f"saved reference row {index} conflicts with the legacy cache")
    if not args.merge:
        score_shard(args, rows, data_args, scores, done)
        if args.shard is not None:
            return
        done = read_completed(args.work_dir, rows)
    scores = ordered_scores(done, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_reference_cache(args.out, rows, scores, contract, legacy_import=legacy_import, generation=generation)
    print(f"Reference cache: {len(scores)} rows saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
