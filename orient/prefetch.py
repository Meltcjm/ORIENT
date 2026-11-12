"""Prepare the next stage's CPU inputs while the current stage scores and trains."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import warnings

from .acceleration import ExactCLIPProcessor
from .data import has_image
from .scoring_inputs import PreparedScoringInputs, ScoringInputCache


def _prepare(rows, tokenizer, data_args, image_folder, cache):
    args = copy(data_args)
    args.image_folder = image_folder
    if hasattr(args, "image_processor") and not isinstance(args.image_processor, ExactCLIPProcessor):
        args.image_processor = ExactCLIPProcessor(args.image_processor)
    for multimodal in (True, False):
        selected = [row for row in rows if has_image(row) is multimodal]
        if selected:
            args.is_multimodal = multimodal
            dataset = PreparedScoringInputs(selected, tokenizer, args, cache)
            for index in range(len(selected)):
                dataset[index]


class OnlineInputPrefetch:
    def __init__(self, enabled, max_bytes=4 * 1024**3):
        if max_bytes < 0:
            raise ValueError("prefetch cache budget must be nonnegative")
        self.enabled, self.max_bytes = bool(enabled), int(max_bytes)
        self.executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="orient-input")
                         if self.enabled else None)
        self.pending = None

    def submit(self, stage, rows, tokenizer, data_args, image_folder):
        if not self.enabled:
            return
        if self.pending is not None:
            raise RuntimeError("input prefetch already has a pending stage")
        cache = ScoringInputCache(self.max_bytes)
        try:
            future = self.executor.submit(_prepare, rows, tokenizer, data_args, image_folder, cache)
        except Exception as error:
            warnings.warn(f"Input prefetch submission failed ({type(error).__name__}); using synchronous inputs.")
            return
        self.pending = int(stage), cache, future

    def take(self, stage):
        if self.pending is None:
            return None
        expected, cache, future = self.pending
        if expected != int(stage):
            raise RuntimeError(f"prefetched stage {expected} does not match requested stage {stage}")
        self.pending = None
        try:
            future.result()
        except Exception as error:
            warnings.warn(f"Input prefetch failed ({type(error).__name__}); using synchronous inputs.")
            return None
        return cache

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.pending = None
