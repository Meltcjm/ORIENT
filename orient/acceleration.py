"""Reuse immutable scoring work with native fallbacks for unsupported backends."""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F


class ExactCLIPProcessor:
    """The native 4.37.2 transforms for padded, square RGB CLIP-336 images."""
    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def preprocess(self, images, **kwargs):
        import transformers
        from PIL import Image
        from transformers import CLIPImageProcessor
        p = self.processor
        if (transformers.__version__ == '4.37.2' and type(p) is CLIPImageProcessor
                and not any(name in p.__dict__ for name in
                            ('preprocess', 'resize', 'center_crop', 'rescale', 'normalize'))
                and isinstance(images, Image.Image) and images.mode == 'RGB'
                and images.width == images.height and images.width > 3
                and kwargs == {'return_tensors': 'pt'}
                and p.size == {'shortest_edge': 336}
                and p.crop_size == {'height': 336, 'width': 336}
                and p.do_resize and p.do_center_crop and p.do_rescale
                and p.do_normalize and p.do_convert_rgb
                and np.asarray(p.image_mean).shape == (3,)
                and np.asarray(p.image_std).shape == (3,)):
            pixels = np.asarray(images.resize((336, 336), resample=p.resample))
            pixels = (pixels * p.rescale_factor).astype(np.float32)
            pixels = (pixels - np.asarray(p.image_mean, dtype=pixels.dtype)) / np.asarray(p.image_std, dtype=pixels.dtype)
            return {'pixel_values': torch.from_numpy(pixels.transpose(2, 0, 1)[None].copy())}
        return p.preprocess(images, **kwargs)


@contextmanager
def reuse_unpadding(model):
    """Compute native prefill mask indices once, shared by this model's layers."""
    import transformers
    from transformers.models.llama import modeling_llama as llama
    attention = [m for m in model.modules() if type(m) is llama.LlamaFlashAttention2]
    if (transformers.__version__ != '4.37.2' or not attention
            or not hasattr(llama, 'index_first_axis')
            or any('_upad_input' in m.__dict__ for m in attention)):
        yield
        return
    cached_mask, cached_version, metadata = None, None, None
    changed = []
    try:
        for attn in attention:
            original = attn._upad_input

            def unpad(this, query, key, value, mask, query_length, *, original=original):
                nonlocal cached_mask, cached_version, metadata
                if query_length != key.shape[1]:
                    return original(query, key, value, mask, query_length)
                version = mask._version
                if mask is not cached_mask or version != cached_version:
                    metadata = llama._get_unpad_data(mask)
                    cached_mask, cached_version = mask, version
                indices, cumulative, maximum = metadata
                batch, sequence, heads, dim = key.shape
                k = llama.index_first_axis(key.reshape(batch * sequence, heads, dim), indices)
                v = llama.index_first_axis(value.reshape(batch * sequence, heads, dim), indices)
                q = llama.index_first_axis(query.reshape(batch * sequence, this.num_heads, dim), indices)
                return q, k, v, indices, (cumulative, cumulative), (maximum, maximum)

            attn._upad_input = MethodType(unpad, attn)
            changed.append(attn)
        yield
    finally:
        for attn in changed:
            del attn._upad_input


def _copy_layout(value):
    # A contiguous clone of sliced CLIP patches can select a different bf16 GEMM.
    size = value.storage_offset() + 1 + sum((s - 1) * t for s, t in zip(value.shape, value.stride()))
    storage = torch.empty(size, dtype=value.dtype, device=value.device)
    result = storage.as_strided(value.shape, value.stride(), value.storage_offset())
    return result.copy_(value)


def answer_sliced_batch_nll(model, batch, device):
    """Project only supervised answer positions; unsupported heads use native scoring."""
    from .scoring import _batch_nll
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    base = model.get_base_model() if callable(getattr(model, 'get_base_model', None)) else model
    if (type(base) is not LlavaLlamaForCausalLM or getattr(base, 'pretraining_tp', 1) != 1
            or getattr(base.config, 'pretraining_tp', 1) != 1):
        return _batch_nll(model, batch, device)
    from llava.constants import IGNORE_INDEX
    inputs = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in batch.items() if key != 'images'}
    labels = inputs.pop('labels')
    images = batch.get('images')
    if torch.is_tensor(images):
        images = images.to(device=device, dtype=next(base.parameters()).dtype)
    elif images is not None:
        images = [image.to(device=device, dtype=next(base.parameters()).dtype) for image in images]
    with torch.no_grad():
        if images is not None and hasattr(base, 'prepare_inputs_labels_for_multimodal'):
            input_ids, position_ids, attention_mask, _, inputs_embeds, labels = base.prepare_inputs_labels_for_multimodal(
                inputs['input_ids'], inputs.get('position_ids'), inputs['attention_mask'], None, labels, images,
                inputs.get('image_sizes'))
            inputs = {'input_ids': input_ids if inputs_embeds is None else None,
                      'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask,
                      'position_ids': position_ids}
        inputs['use_cache'] = False
        hidden = base.model(**inputs)[0][:, :-1]
        targets = labels[:, 1:].to(hidden.device).contiguous()
        keep = targets.ne(IGNORE_INDEX)
        counts = keep.sum(-1)
        rows, cols = keep.nonzero(as_tuple=True)
        if rows.numel():
            logits = base.lm_head(hidden[rows, cols]).float()
            losses = F.cross_entropy(logits, targets[rows, cols], reduction='none')
            per_token = losses.new_zeros(targets.shape)
            per_token[rows, cols] = losses
            values = per_token.sum(-1) / counts.clamp_min(1)
        else:
            values = hidden.new_zeros((len(counts),), dtype=torch.float32)
        values, counts = values.detach().cpu().tolist(), counts.detach().cpu().tolist()
    return [(float(value) if count else None, int(count)) for value, count in zip(values, counts)]


class FrozenVisionBatches:
    """Bounded cache of whole native batches, before the live mm_projector."""
    def __init__(self, max_bytes=512 * 1024**2):
        if max_bytes < 0:
            raise ValueError('vision cache budget must be nonnegative')
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.state = None

    def _clear(self):
        self.entries.clear()
        self.bytes = 0
        self.state = None

    @contextmanager
    def reuse(self, model, cpu_images, cache_outputs=True):
        from llava.model.llava_arch import LlavaMetaForCausalLM
        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
        from transformers import CLIPVisionModel
        from torch.nn.modules import module as module_impl
        tower = model.get_vision_tower() if hasattr(model, 'get_vision_tower') else None
        native = (torch.is_tensor(cpu_images) and cpu_images.device.type == 'cpu' and cpu_images.ndim == 4
                  and type(tower) is CLIPVisionTower and tower.is_loaded
                  and type(tower.vision_tower) is CLIPVisionModel
                  and getattr(tower.forward, '__func__', None) is CLIPVisionTower.forward
                  and getattr(tower.feature_select, '__func__', None) is CLIPVisionTower.feature_select
                  and getattr(getattr(model, 'encode_images', None), '__func__', None) is LlavaMetaForCausalLM.encode_images
                  and getattr(getattr(model, 'prepare_inputs_labels_for_multimodal', None), '__func__', None)
                      is LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal)
        if native:
            modules = list(tower.modules())
            tensors = [*tower.named_parameters(), *tower.named_buffers()]
            native = (bool(tensors) and not any(p.requires_grad for p in tower.parameters())
                      and not module_impl._global_forward_hooks and not module_impl._global_forward_pre_hooks
                      and not any(m.training or m._forward_hooks or m._forward_pre_hooks
                                  or hasattr(m, '_hf_hook') or 'forward' in m.__dict__ for m in modules))
        state = None
        if native:
            try:
                state = (id(tower), tower.select_layer, tower.select_feature,
                    json.dumps(tower.config.to_dict(), sort_keys=True),
                    tuple((id(m), type(m), getattr(m.forward, '__func__', None), getattr(m, 'eps', None)) for m in modules),
                    tuple((n, id(p), p._version, p.device, p.dtype) for n, p in tensors),
                    torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype(),
                    torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype(),
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                    torch.backends.mkldnn.enabled,
                    torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision())
            except RuntimeError:
                pass  # Inference tensors have no mutation counter; use the native forward.
        if state is None:
            if cpu_images is not None:
                self._clear()
            yield
            return
        if state != self.state:
            self._clear()
            self.state = state
        digest = (hashlib.sha256(cpu_images.contiguous().view(torch.uint8).numpy()).hexdigest()
                  if cache_outputs else None)
        original = tower.forward

        def forward(images):
            if not torch.is_tensor(images) or images.shape != cpu_images.shape:
                return original(images)
            key = (digest, cpu_images.dtype, cpu_images.stride(), images.shape,
                   images.device, images.dtype, images.stride()) if cache_outputs else None
            if key is not None and key in self.entries:
                self.hits += 1
                value, size = self.entries.pop(key)
                self.entries[key] = value, size
                return _copy_layout(value)
            value = self._selected_layer_forward(tower, images)
            if key is not None:
                self.misses += 1
            if key is not None and torch.is_tensor(value) and value.numel():
                size = (value.storage_offset() + 1
                        + sum((s - 1) * t for s, t in zip(value.shape, value.stride()))) * value.element_size()
                if size <= self.max_bytes:
                    while self.entries and self.bytes + size > self.max_bytes:
                        _, (_, evicted) = self.entries.popitem(last=False)
                        self.bytes -= evicted
                    self.entries[key] = _copy_layout(value), size
                    self.bytes += size
            return value

        tower.forward = forward
        try:
            yield
        finally:
            del tower.forward

    @staticmethod
    def _selected_layer_forward(tower, images):
        """Run only the frozen CLIP layers needed by LLaVA's selected hidden state."""
        import transformers
        from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPEncoderLayer, CLIPVisionTransformer
        core = tower.vision_tower.vision_model
        layers = core.encoder.layers
        target = tower.select_layer if tower.select_layer >= 0 else len(layers) + 1 + tower.select_layer
        if (transformers.__version__ != '4.37.2' or type(core) is not CLIPVisionTransformer
                or type(core.encoder) is not CLIPEncoder
                or any(type(layer) is not CLIPEncoderLayer for layer in layers)
                or not 0 <= target <= len(layers)
                or tower.select_feature not in ('patch', 'cls_patch')):
            outputs = tower.vision_tower(images.to(device=tower.device, dtype=tower.dtype),
                                         output_hidden_states=True)
            return tower.feature_select(outputs).to(images.dtype)
        hidden = core.pre_layrnorm(core.embeddings(images.to(device=tower.device, dtype=tower.dtype)))
        for layer in layers[:target]:
            hidden = layer(hidden, attention_mask=None, causal_attention_mask=None,
                           output_attentions=False)[0]
        if tower.select_feature == 'patch':
            hidden = hidden[:, 1:]
        return hidden.to(images.dtype)


class ScoringRuntime:
    """Retain only frozen vision outputs across scoring calls."""
    def __init__(self, answer_sliced=False):
        self.answer_sliced = bool(answer_sliced)
        self.vision = FrozenVisionBatches()

    def batch(self, model, tensors, device, capture_hidden=False, cache_vision=False):
        from .scoring import _batch_nll, final_state_batch_nll
        base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
        scorer = answer_sliced_batch_nll if self.answer_sliced else _batch_nll
        with self.vision.reuse(base, tensors.get("images"), cache_outputs=cache_vision):
            if capture_hidden:
                scores, hidden = final_state_batch_nll(model, tensors, device, scorer)
                return [(nll, count, vector) for (nll, count), vector in zip(scores, hidden)]
            return [(nll, count, None) for nll, count in scorer(model, tensors, device)]
