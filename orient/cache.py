"""Reference-loss cache with exact source and scoring bindings."""
import json
from pathlib import Path
import math
import pickle

from .data import file_sha256, has_image, source_rows_sha256, write_json_exclusive


SCORING_MODE = "orient-paired-answer-v1"


def scoring_contract(model_args, data_args, training_args, reference_model):
    paths = {name: Path(path).resolve() for name, path in (
        ("base", model_args.model_name_or_path), ("reference", reference_model),
        ("vision", model_args.vision_tower))}
    hashes = {path: checkpoint_hashes(path) for path in dict.fromkeys(paths.values())}
    return {"model_name_or_path": str(Path(model_args.model_name_or_path).resolve()),
            "reference_model": str(Path(reference_model).resolve()),
            "vision_tower": str(Path(model_args.vision_tower).resolve()),
            "version": model_args.version, "model_max_length": int(training_args.model_max_length),
            "image_aspect_ratio": data_args.image_aspect_ratio,
            "mm_use_im_start_end": bool(model_args.mm_use_im_start_end),
            "dtype": "bfloat16" if training_args.bf16 else "float16" if training_args.fp16 else "float32",
            "checkpoint_sha256": {name: hashes[path] for name, path in paths.items()}}


def checkpoint_hashes(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in (
        ".json", ".bin", ".safetensors", ".model", ".txt"))
    if not files or not any(p.suffix in (".bin", ".safetensors") for p in files):
        raise ValueError(f"expected a local checkpoint directory with weights: {directory}")
    return {p.name: file_sha256(p) for p in files}


def validate_scores(scores, rows):
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise ValueError("reference cache must cover every ordered source row")
    for score, row in zip(scores, rows):
        if not isinstance(score, dict) or score.get("has_image") is not has_image(row):
            raise ValueError("reference image flags do not match source rows")
        channels = ("T", "V") if has_image(row) else ("T",)
        for channel in channels:
            count, value = score.get(f"answer_tokens_{channel}"), score.get(f"ell_{channel}")
            if type(count) is not int or count < 0:
                raise ValueError("invalid supervised answer-token count")
            if count == 0:
                if value is not None or score.get("invalid_reason") != "no_supervised_answer_tokens":
                    raise ValueError("undefined NLL must be marked explicitly")
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid reference NLL")
        if score.get("invalid_reason") and not any(score[f"answer_tokens_{c}"] == 0 for c in channels):
            raise ValueError("invalid NLL marker conflicts with supervised token counts")
        if has_image(row) and score['answer_tokens_T'] != score['answer_tokens_V']:
            raise ValueError("T/V must supervise the same answer-token prefix")


def write_reference_cache(path, rows, scores, contract, legacy_import=None, generation=None):
    validate_scores(scores, rows)
    if Path(path).exists():
        raise FileExistsError("reference cache already exists")
    cache = {"schema": "orient-reference-v1", "scoring_mode": SCORING_MODE,
             "source_rows_sha256": source_rows_sha256(rows),
             "scoring_contract": contract, "scores": scores}
    if legacy_import is not None:
        cache["legacy_import"] = legacy_import
    if generation is not None:
        cache["generation"] = generation
    write_json_exclusive(path, cache)


def load_reference_cache(path, rows, contract=None, *, image_folder=None, return_contract=False):
    cache = json.loads(Path(path).read_text())
    if (cache.get("schema") != "orient-reference-v1" or cache.get("scoring_mode") != SCORING_MODE
            or cache.get("source_rows_sha256") != source_rows_sha256(rows)
            or not isinstance(cache.get("scoring_contract"), dict)):
        raise ValueError("reference cache is unbound or belongs to different ordered rows")
    if contract is not None and cache["scoring_contract"] != contract:
        raise ValueError("reference model or preprocessing differs from cache")
    if image_folder is not None:
        generation = cache.get("generation")
        bound_folder = generation.get("image_folder") if isinstance(generation, dict) else None
        if (not isinstance(bound_folder, str) or not bound_folder
                or Path(bound_folder).resolve() != Path(image_folder).resolve()):
            raise ValueError("reference cache image folder is missing or differs; regenerate the cache")
    validate_scores(cache.get("scores"), rows)
    return (cache["scores"], cache["scoring_contract"]) if return_contract else cache["scores"]


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("legacy cache may contain only primitive values, not Python objects")


def load_legacy_reference(path, rows, contract):
    """Explicitly import path-bound scores; missing or unaligned rows need rescoring."""
    with Path(path).open("rb") as handle:
        cache = _PrimitiveUnpickler(handle).load()
    old_contract = {key: value for key, value in contract.items() if key != "checkpoint_sha256"}
    if (not isinstance(cache, dict) or cache.get("scoring_mode") != "paired_tv_v1"
            or cache.get("source_rows_sha256") != source_rows_sha256(rows)
            or cache.get("scoring_contract") != old_contract):
        raise ValueError("legacy cache belongs to different ordered rows, models or preprocessing")
    indices, values = cache.get("indices"), cache.get("rows")
    if (not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values)
            or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("legacy cache indices must be unique valid source indices")
    scores = [None] * len(rows)
    for index, value in zip(indices, values):
        if not isinstance(value, dict) or value.get("has_image") is not has_image(rows[index]):
            raise ValueError("legacy cache image flags differ from source rows")
        channels = ("T", "V") if has_image(rows[index]) else ("T",)
        score = {"has_image": value["has_image"]}
        for channel in channels:
            count = value.get(f"n_ans_{channel}")
            if type(count) is not int or count < 0:
                raise ValueError("invalid legacy supervised answer-token count")
            score[f"answer_tokens_{channel}"] = count
            score[f"ell_{channel}"] = value.get(f"l{channel}")
        if value.get("invalid_reason"):
            score["invalid_reason"] = value["invalid_reason"]
        if score["has_image"] and score["answer_tokens_T"] != score["answer_tokens_V"]:
            continue
        validate_scores([score], [rows[index]])
        scores[index] = score
    missing = [i for i, score in enumerate(scores) if score is None]
    provenance = {
        "source_sha256": file_sha256(path), "scoring_mode": "paired_tv_v1",
        "scoring_contract_at_generation": old_contract,
        "checkpoint_binding": "legacy paths at generation; content hashes only checked at import",
        "reuse_assumption": "checkpoint and image contents at the bound paths are unchanged",
        "reused_rows": len(rows) - len(missing), "rescored_indices": missing,
    }
    return scores, provenance
