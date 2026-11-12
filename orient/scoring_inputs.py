"""Bounded, CPU-only native scoring inputs; never cache model outputs."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


class ScoringInputCache:
    def __init__(self, max_bytes=512 * 1024**2):
        if max_bytes < 0:
            raise ValueError("input cache budget must be nonnegative")
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.entries = OrderedDict()
        self.file_digests = OrderedDict()

    def get(self, key, prepare):
        if key in self.entries:
            value, size = self.entries.pop(key)
            self.entries[key] = value, size
            return value
        value = prepare()
        tensors = value.values() if isinstance(value, dict) else (value,)
        size = sum(t.numel() * t.element_size() for t in tensors) + len(str(key))
        if size <= self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, (_, evicted) = self.entries.popitem(last=False)
                self.bytes -= evicted
            self.entries[key] = value, size
            self.bytes += size
        return value


class PreparedScoringInputs:
    """Split native token/image transforms so bucketing reuses the exact tokens."""
    def __init__(self, rows, tokenizer, data_args, cache=None):
        from llava import conversation as conversation_lib
        self.rows, self.tokenizer, self.args = rows, tokenizer, data_args
        self.cache = cache if cache is not None else ScoringInputCache(0)
        self.tokens = ScoringInputCache(32 * 1024**2)
        # Recompute the contract per call: callers may mutate tokenizer/template/processor.
        self.contract = _digest({
            "tokenizer_class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
            "vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map,
            "tokenizer_init": tokenizer.init_kwargs, "added_tokens": tokenizer.added_tokens_decoder,
            "tokenizer_flags": {name: getattr(tokenizer, name, None) for name in (
                "add_bos_token", "add_eos_token", "legacy", "truncation_side",
                "clean_up_tokenization_spaces", "split_special_tokens")},
            "tokenizer_model": hashlib.sha256(tokenizer.sp_model.serialized_model_proto()).hexdigest()
                if hasattr(tokenizer, "sp_model") else None,
            "tokenizer_backend": tokenizer.backend_tokenizer.to_str() if getattr(tokenizer, "is_fast", False) else None,
            "max_length": tokenizer.model_max_length, "padding_side": tokenizer.padding_side,
            "conversation": vars(conversation_lib.default_conversation),
            "is_multimodal": data_args.is_multimodal,
            "mm_use_im_start_end": getattr(data_args, "mm_use_im_start_end", False),
            "aspect_ratio": getattr(data_args, "image_aspect_ratio", None),
            "processor": data_args.image_processor.to_dict() if hasattr(data_args, "image_processor") else None,
            "processor_class": (type(data_args.image_processor).__module__ + "." + type(data_args.image_processor).__qualname__)
                if hasattr(data_args, "image_processor") else None,
        })

    def tokenized(self, index):
        from llava.train.train import preprocess, preprocess_multimodal
        def tokenized():
            row = self.rows[index]
            def prepare():
                source = deepcopy([row["conversations"]])
                if "image" in row:
                    source = preprocess_multimodal(source, self.args)
                result = preprocess(source, self.tokenizer, has_image="image" in row)
                return {key: result[key][0] for key in ("input_ids", "labels")}
            return self.cache.get(("tokens", self.contract, _digest(row)), prepare)
        return self.tokens.get(index, tokenized)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from llava.mm_utils import expand2square
        # Collation creates fresh padded tensors; cached tensors are never masked in place.
        result = dict(self.tokenized(index))
        row = self.rows[index]
        if "image" in row:
            path = Path(self.args.image_folder) / row["image"]
            path_key = str(path.resolve())
            stat = path.stat()
            fingerprint = (path_key, stat.st_dev, stat.st_ino, stat.st_size,
                           stat.st_mtime_ns, stat.st_ctime_ns)
            digests = self.cache.file_digests
            raw = None
            if fingerprint in digests:
                digest = digests.pop(fingerprint)
                digests[fingerprint] = digest
            else:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                digests[fingerprint] = digest
                # shortcut: retain 8192 file versions; grow for larger repeated panels.
                if len(digests) > 8192:
                    digests.popitem(last=False)
            image_key = ("image", self.contract, path_key, digest)
            def prepare():
                with Image.open(BytesIO(raw if raw is not None else path.read_bytes())) as source:
                    image = source.convert("RGB")
                processor = self.args.image_processor
                if self.args.image_aspect_ratio == "pad":
                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                return processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            result["image"] = self.cache.get(image_key, prepare)
        elif self.args.is_multimodal:
            crop = self.args.image_processor.crop_size
            result["image"] = torch.zeros(3, crop["height"], crop["width"])
        return result
