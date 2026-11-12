"""Source-bound, atomic reference chunks indexed by global source row."""
import fcntl
import json
from pathlib import Path
from uuid import uuid4

from .cache import SCORING_MODE, validate_scores
from .data import file_sha256, source_rows_sha256, write_json_exclusive


def bind_shards(directory, rows, contract, generation, legacy_import):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = {"schema": "orient-reference-shards-v1", "scoring_mode": SCORING_MODE,
               "source_rows_sha256": source_rows_sha256(rows), "row_count": len(rows),
               "scoring_contract": contract, "generation": generation, "legacy_import": legacy_import}
    with (directory / "binding.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "binding.json"
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError("reference shards belong to different ordered rows, models or generation settings")
        else:
            if any(directory.glob("chunk-*.json")):
                raise ValueError("existing reference chunks lack source binding")
            write_json_exclusive(path, binding)
    return binding


def _validate_chunk(indices, scores, rows):
    if (not isinstance(indices, list) or not indices
            or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("reference chunk indices must be unique valid global indices")
    validate_scores(scores, [rows[i] for i in indices])


def read_completed(directory, rows):
    done = {}
    binding_sha256 = file_sha256(Path(directory) / "binding.json")
    for path in sorted(Path(directory).glob("chunk-*.json")):
        chunk = json.loads(path.read_text())
        if not isinstance(chunk, dict) or chunk.get("scoring_mode") != SCORING_MODE:
            raise ValueError(f"invalid reference chunk: {path.name}")
        if chunk.get("binding_sha256") != binding_sha256:
            raise ValueError(f"reference chunk belongs to a different binding: {path.name}")
        indices, scores = chunk.get("indices"), chunk.get("scores")
        _validate_chunk(indices, scores, rows)
        for index, score in zip(indices, scores):
            if index in done and done[index] != score:
                raise ValueError(f"conflicting reference scores for global index {index}")
            done[index] = score
    return done


def save_chunk(directory, indices, scores, rows):
    _validate_chunk(indices, scores, rows)
    path = Path(directory) / f"chunk-{uuid4().hex}.json"
    write_json_exclusive(path, {"scoring_mode": SCORING_MODE, "indices": indices, "scores": scores,
                               "binding_sha256": file_sha256(Path(directory) / "binding.json")})


def ordered_scores(done, rows):
    missing = [i for i in range(len(rows)) if i not in done]
    if missing:
        raise ValueError(f"{len(missing)} reference rows still unscored (first global indices: {missing[:5]})")
    return [done[i] for i in range(len(rows))]
