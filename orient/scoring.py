"""Native answer-token NLLs and live pre-answer hidden states."""
import torch

from .data import has_image
from .scoring_inputs import PreparedScoringInputs
from .acceleration import ExactCLIPProcessor, ScoringRuntime, reuse_unpadding


def final_state_batch_nll(model, batch, device, score_batch):
    """Read the pre-answer state from the same decoder forward used for NLL."""
    base = model.get_base_model() if callable(getattr(model, 'get_base_model', None)) else model
    decoder = base.get_model() if callable(getattr(base, 'get_model', None)) else None
    if decoder is None or not hasattr(decoder, 'norm'):
        raise ValueError('online semantic coverage requires a decoder with a final norm')
    context = dict(labels=batch['labels'], attention_mask=batch.get('attention_mask'))
    captured = []
    prepared = []

    def capture(_module, _args, hidden):
        labels = context['labels'].to(hidden.device)
        supervised = labels.ne(-100)
        first = supervised.long().argmax(1)
        if not bool(supervised.any(1).all()) or bool((first == 0).any()):
            raise ValueError('semantic feature has no valid input position before its first answer')
        positions = first - 1
        attention = context['attention_mask']
        if attention is not None and not bool(attention.to(hidden.device)[torch.arange(len(hidden), device=hidden.device), positions].all()):
            raise ValueError('pre-answer semantic position is padding')
        vectors = hidden[torch.arange(len(hidden), device=hidden.device), positions].detach().float()
        norms = vectors.norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(vectors).all()) or bool((norms <= 1e-8).any()):
            raise ValueError('nonfinite or empty pre-answer semantic feature')
        captured.append((vectors / norms).to(dtype=torch.float16).cpu().numpy())

    handle = decoder.norm.register_forward_hook(capture)
    try:
        for owner in (model,) if model is base else (model, base):
            original = getattr(owner, 'prepare_inputs_labels_for_multimodal', None)
            if original is None:
                continue
            def prepare(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                if result[5] is not None:
                    context.update(labels=result[5], attention_mask=result[2])
                return result
            prepared.append((owner, original, 'prepare_inputs_labels_for_multimodal' in owner.__dict__))
            owner.prepare_inputs_labels_for_multimodal = prepare
        result = score_batch(model, batch, device)
        if len(captured) != 1 or len(captured[0]) != len(result):
            raise RuntimeError('semantic capture requires exactly one aligned decoder forward')
        return result, captured[0]
    finally:
        handle.remove()
        for owner, original, own in reversed(prepared):
            if own:
                owner.prepare_inputs_labels_for_multimodal = original
            else:
                del owner.prepare_inputs_labels_for_multimodal

def _batch_nll(model, batch, device) -> list[tuple[float | None, int]]:
    import torch
    import torch.nn.functional as F
    inputs = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in batch.items() if key != 'images'}
    labels = inputs.pop('labels')
    images = batch.get('images')
    if torch.is_tensor(images):
        images = images.to(device=device, dtype=next(model.parameters()).dtype)
    elif images is not None:
        images = [image.to(device=device, dtype=next(model.parameters()).dtype) for image in images]
    with torch.no_grad():
        if images is not None and hasattr(model, 'prepare_inputs_labels_for_multimodal'):
            (input_ids, position_ids, attention_mask, _, inputs_embeds, labels) = model.prepare_inputs_labels_for_multimodal(inputs['input_ids'], inputs.get('position_ids'), inputs['attention_mask'], None, labels, images, inputs.get('image_sizes'))
            inputs = {'input_ids': input_ids if inputs_embeds is None else None, 'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask, 'position_ids': position_ids}
        inputs['use_cache'] = False
        logits = model(**inputs).logits[:, :-1].float().contiguous()
    target = labels[:, 1:].to(logits.device).contiguous()
    mask = target.ne(-100)
    losses = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100, reduction='none').view_as(target)
    counts = mask.sum(-1)
    per_row = losses.sum(-1) / counts.clamp_min(1)
    return [(float(value) if count else None, int(count)) for (value, count) in zip(per_row.detach().cpu(), counts.cpu())]

def _text_counterfactual(batch, pad_token_id):
    """Keep V token positions while removing only the visual input."""
    import torch
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    result = {key: value.clone() if torch.is_tensor(value) else value for (key, value) in batch.items() if key != 'images'}
    image_mask = result['input_ids'].eq(IMAGE_TOKEN_INDEX)
    result['labels'][image_mask] = IGNORE_INDEX
    result['input_ids'][image_mask] = int(pad_token_id)
    if 'attention_mask' in result:
        result['attention_mask'][image_mask] = 0
    if 'images' in batch:
        result['images'] = None
    return result

def _length_order(rows, tokenizer, data_args, model, bucket, prepared=None):
    from copy import deepcopy
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.train.train import preprocess, preprocess_multimodal
    order = list(range(len(rows)))
    if bucket and len(order) > 1:
        vision_tower = model.get_vision_tower() if hasattr(model, 'get_vision_tower') else None
        image_tokens = max(0, int(getattr(vision_tower, 'num_patches', 1)) - 1)
        keyed = []
        for (index, row) in enumerate(rows):
            image = has_image(row)
            if prepared is None:
                source = deepcopy([row['conversations']])
                if image:
                    source = preprocess_multimodal(source, data_args)
                ids = preprocess(source, tokenizer, has_image=image)['input_ids'][0]
            else:
                ids = prepared.tokenized(index)['input_ids']
            expanded = int(ids.eq(IMAGE_TOKEN_INDEX).sum()) * image_tokens if image else 0
            keyed.append((len(ids) + expanded, index))
        keyed.sort()
        order = [index for (_, index) in keyed]
    return order

def _score_native_image_rows(model, tokenizer, rows: list[dict], data_args, batch_size: int, channels: tuple[str, ...], bucket: bool, input_cache, capture_hidden, runtime, cache_vision) -> list[dict]:
    """Supervise the same answer prefix after visual sequence expansion."""
    import torch
    from copy import copy
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.train.train import DataCollatorForSupervisedDataset
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    native_args = copy(data_args)
    native_args.is_multimodal = True
    dataset = PreparedScoringInputs(rows, tokenizer, native_args, input_cache)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    results = [{'has_image': has_image(row)} for row in rows]
    order = _length_order(rows, tokenizer, native_args, model, bucket, dataset)
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for start in range(0, len(order), batch_size):
        if start and start // batch_size % 16 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        indices = order[start:start + batch_size]
        instances = [dataset[index] for index in indices]
        batch = collator(instances)
        image_mask = batch['input_ids'].eq(IMAGE_TOKEN_INDEX)
        batch['labels'][image_mask] = IGNORE_INDEX
        visual_values = runtime.batch(model, batch, device, capture_hidden, cache_vision)
        if 'V' in channels:
            for (index, (nll, count, hidden)) in zip(indices, visual_values):
                results[index].update(ell_V=nll, answer_tokens_V=count)
                if hidden is not None:
                    results[index]['h'] = hidden
        if 'T' in channels:
            text_batch = _text_counterfactual(batch, pad_token_id)
            align_answer_prefix(text_batch['labels'], [value[1] for value in visual_values])
            text_values = runtime.batch(model, text_batch, device)
            for (index, (nll, count, _)) in zip(indices, text_values):
                results[index].update(ell_T=nll, answer_tokens_T=count)
    return results

def _score_native_text_rows(model, tokenizer, rows: list[dict], data_args, batch_size: int, bucket: bool, input_cache, capture_hidden, runtime):
    """Score rows that have no image token through native text preprocessing."""
    from copy import copy
    from llava.train.train import DataCollatorForSupervisedDataset
    native_args = copy(data_args)
    native_args.is_multimodal = False
    dataset = PreparedScoringInputs(rows, tokenizer, native_args, input_cache)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    order = _length_order(rows, tokenizer, native_args, model, bucket, dataset)
    device = next(model.parameters()).device
    values = [None] * len(rows)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch_values = runtime.batch(model, collator([dataset[index] for index in indices]), device, capture_hidden)
        for (index, value) in zip(indices, batch_values):
            values[index] = value
    return values

@torch.no_grad()
def score_rows(model, tokenizer, rows: list[dict], image_folder: str, data_args, batch_size: int=4, channels: tuple[str, ...]=('T', 'V'), bucket: bool=True, input_cache=None, capture_hidden=False, runtime=None, feedback_only=False) -> list[dict]:
    """Return paired native text/visual answer NLLs in input order."""
    if batch_size <= 0 or not channels or (not set(channels) <= {'T', 'V'}):
        raise ValueError('batch_size must be positive and channels must contain T or V')
    from copy import copy
    data_args = copy(data_args)
    data_args.image_folder = image_folder
    if hasattr(data_args, 'image_processor') and not isinstance(data_args.image_processor, ExactCLIPProcessor):
        data_args.image_processor = ExactCLIPProcessor(data_args.image_processor)
    runtime = runtime if runtime is not None else ScoringRuntime()
    was_training = model.training
    was_use_cache = getattr(model.config, 'use_cache', None)
    model.eval()
    if was_use_cache is not None:
        model.config.use_cache = False
    image_indices = [i for (i, row) in enumerate(rows) if has_image(row)]
    text_indices = [i for (i, row) in enumerate(rows) if not has_image(row)]
    image_rows = [rows[i] for i in image_indices]
    text_rows = [rows[i] for i in text_indices]
    results = [{'has_image': has_image(row)} for row in rows]
    try:
        with reuse_unpadding(model):
            if image_rows:
                image_channels = ('V',) if feedback_only else channels
                for (index, score) in zip(image_indices, _score_native_image_rows(model, tokenizer, image_rows, data_args, batch_size, image_channels, bucket, input_cache, capture_hidden, runtime, feedback_only)):
                    results[index].update(score)
            if 'T' in channels and text_rows:
                for (index, (nll, count, hidden)) in zip(text_indices, _score_native_text_rows(model, tokenizer, text_rows, data_args, batch_size, bucket, input_cache, capture_hidden, runtime)):
                    results[index].update(ell_T=nll, answer_tokens_T=count)
                    if hidden is not None:
                        results[index]["h"] = hidden
        for result in results:
            if any((result.get(f'answer_tokens_{channel}') == 0 for channel in channels)):
                result['invalid_reason'] = 'no_supervised_answer_tokens'
        return results
    finally:
        if was_use_cache is not None:
            model.config.use_cache = was_use_cache
        if was_training:
            model.train()


def align_answer_prefix(labels, visual_counts):
    """Right truncation after image expansion retains an answer-token prefix."""
    if len(labels) != len(visual_counts):
        raise ValueError('visual counts and text rows differ')
    for row, count in zip(labels, visual_counts):
        positions = row[1:].ne(-100).nonzero(as_tuple=True)[0] + 1
        if not 0 <= count <= len(positions):
            raise ValueError('visual targets are not a prefix of text targets')
        row[positions[count:]] = -100
