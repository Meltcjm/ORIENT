"""Collect scoring shards and propagate controller failures to every rank."""
from datetime import timedelta
import traceback
import torch


class Collective:
    def __init__(self):
        dist = torch.distributed
        self.enabled = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.enabled else 0
        self.world_size = dist.get_world_size() if self.enabled else 1
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=600)) if self.enabled else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.group is not None:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group(self.group)
            self.group = None

    def main(self, operation):
        payload = [None]
        if self.rank == 0:
            try:
                payload[0] = (operation(), None)
            except Exception:
                payload[0] = (None, traceback.format_exc())
        if self.enabled:
            torch.distributed.broadcast_object_list(payload, src=0, group=self.group)
        result, error = payload[0]
        if error:
            raise RuntimeError(f"ORIENT controller failed:\n{error}")
        return result

    def score(self, indices, operation):
        positions = list(range(self.rank, len(indices), self.world_size))
        try:
            local = (positions, operation([indices[i] for i in positions]), None)
        except Exception:
            local = (positions, [], traceback.format_exc())
        if self.enabled:
            shards = [None] * self.world_size if self.rank == 0 else None
            torch.distributed.gather_object(local, shards, dst=0, group=self.group)
        else:
            shards = [local]
        result = None

        def assemble():
            nonlocal result
            result = merge_shards(shards, len(indices))

        self.main(assemble)
        return result


def merge_shards(shards, size):
    result = [None] * size
    for rank, (positions, rows, error) in enumerate(shards):
        if error:
            raise RuntimeError(f"scoring rank {rank} failed:\n{error}")
        if len(positions) != len(rows):
            raise ValueError("scoring positions and values differ")
        for position, row in zip(positions, rows):
            if (type(position) is not int or not 0 <= position < size
                    or result[position] is not None or row is None):
                raise ValueError("invalid or duplicate scoring position")
            result[position] = row
    if any(row is None for row in result):
        raise ValueError("scoring shards do not cover the requested rows")
    return result
