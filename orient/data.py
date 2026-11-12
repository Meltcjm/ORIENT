"""Source identity, fixed feedback sets, and the stage schedule in Eq. (4)."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile


def has_image(row):
    return "image" in row and row["image"] not in (None, "")


def source_rows_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_json_exclusive(path, value):
    """Publish a durable complete file without overwriting an existing result."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink()
            raise
    try:
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink()


def load_rows(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("data must be a nonempty JSON list")
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("conversations"), list)
                or not row["conversations"]):
            raise ValueError("each sample must contain conversations")
        if "image" in row and (not isinstance(row["image"], str) or not row["image"]):
            raise ValueError("text-only samples must omit the image field")
        for turn in row["conversations"]:
            if (not isinstance(turn, dict) or not isinstance(turn.get("value"), str)
                    or turn.get("from") not in ("human", "gpt")):
                raise ValueError("native LLaVA conversations require human/gpt roles")
    return rows


def load_metadata(path, rows):
    metadata = json.loads(Path(path).read_text())
    if (metadata.get("schema") != "orient-metadata-v1"
            or metadata.get("source_rows_sha256") != source_rows_sha256(rows)):
        raise ValueError("metadata is not bound to these ordered source rows")
    tasks = metadata.get("tasks")
    if (not isinstance(tasks, list) or len(tasks) != len(rows)
            or any(not isinstance(task, str) or not task for task in tasks)):
        raise ValueError("task labels must align with every source row")
    names = metadata.get("task_names")
    if (not isinstance(names, list) or any(not isinstance(t, str) for t in names)
            or len(names) != len(set(names)) or set(names) != set(tasks)):
        raise ValueError("task_names must list each task exactly once")
    for name in names:
        if len({has_image(row) for row, task in zip(rows, tasks) if task == name}) != 1:
            raise ValueError("image-grounded and text-only samples must use separate tasks")
    feedback = metadata.get("feedback_indices")
    reserved = metadata.get("reserved_indices")
    for indices in (feedback, reserved):
        if (not isinstance(indices, list) or not indices
                or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("feedback/reserved indices must be unique valid source indices")
    if not set(feedback) <= set(reserved) or len(reserved) >= len(rows):
        raise ValueError("feedback must be reserved and leave a nonempty candidate pool")
    if {tasks[i] for i in feedback} != set(names):
        raise ValueError("fixed feedback must cover every task")
    return metadata


def feedback_groups(metadata):
    # Preserve native feedback batch boundaries; the reward remains a task macro mean.
    groups = [("text_only",), ("conversation", "description", "reasoning"),
              ("vqav2", "aokvqa", "gqa"), ("refcoco", "visual_genome"), ("ocr_vqa", "textcaps")]
    known = {task for group in groups for task in group}
    groups.extend((task,) for task in metadata["task_names"] if task not in known)
    indices = [[i for i in metadata["feedback_indices"] if metadata["tasks"][i] in group]
               for group in groups]
    return [group for group in indices if group]


@dataclass(frozen=True)
class Stage:
    t: int
    P_t: list[int]
    K_t: int
    offset: int
    updates: int


def plan_stages(candidate_indices, rho, B, n, seed):
    if (not math.isfinite(rho) or not 0 < rho <= 1
            or type(B) is not int or B <= 0 or type(n) is not int or n <= 0):
        raise ValueError("rho must be in (0, 1]; B and n must be positive integers")
    indices = list(candidate_indices)
    if (any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("candidate indices must be unique nonnegative integers")
    random.Random(seed).shuffle(indices)
    rate = Fraction(str(rho))
    K = int(len(indices) * rate // B) * B
    if K < B:
        raise ValueError("selection budget is smaller than one global SFT batch")
    stages, selected, start = [], 0, 0
    while selected < K:
        K_t = min(n * B, K - selected)
        stop = len(indices) if selected + K_t == K else math.ceil(Fraction(selected + K_t) / rate)
        P_t = indices[start:stop]
        if len(P_t) < K_t:
            raise ValueError("candidate stage is smaller than its budget")
        stages.append(Stage(len(stages), P_t, K_t, selected, K_t // B))
        selected += K_t
        start = stop
    return stages
