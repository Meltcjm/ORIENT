"""Create source-bound task labels and a fixed task-balanced feedback split."""
import argparse
import json
from pathlib import Path
import random

from .data import has_image, load_metadata, load_rows, source_rows_sha256, write_json


LLAVA_TASK_NAMES = ("conversation", "description", "reasoning", "vqav2", "aokvqa",
                    "refcoco", "visual_genome", "gqa", "ocr_vqa", "textcaps", "text_only")


def load_task_labels(rows, task_root=None):
    if task_root is None:
        tasks = [row.get("task") for row in rows]
        if any(not isinstance(task, str) or not task for task in tasks):
            raise ValueError("provide a task field on every row or --task-root")
        return tasks, list(dict.fromkeys(tasks))
    wanted = {(str(row.get("id", "")), row.get("image")) for row in rows if has_image(row)}

    def key(row):
        return (str(row.get("id", "")), row.get("image"),
                tuple((turn.get("from"), turn.get("value")) for turn in row.get("conversations", [])))

    lookup = {}
    for task_id, task in enumerate(LLAVA_TASK_NAMES[:-1], 1):
        path = Path(task_root) / f"file{task_id}_665.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if (str(row.get("id", "")), row.get("image")) not in wanted:
                continue
            if not row.get("conversations"):
                raise ValueError("task mapping requires full conversations, not ambiguous image IDs")
            identity = key(row)
            if identity in lookup and lookup[identity] != task:
                raise ValueError("the same conversation belongs to conflicting tasks")
            lookup[identity] = task
    tasks = [lookup.get(key(row)) if has_image(row) else "text_only" for row in rows]
    if None in tasks:
        raise ValueError("some image-grounded conversations are missing from task files")
    return tasks, [name for name in LLAVA_TASK_NAMES if name in tasks]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--task-root", type=Path)
    parser.add_argument("--feedback-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; choose a new metadata file")
    if args.feedback_per_task <= 0:
        parser.error("feedback-per-task must be positive")
    rows = load_rows(args.data_path)
    tasks, names = load_task_labels(rows, args.task_root)
    rng = random.Random(args.seed)
    feedback = []
    for name in names:
        members = [i for i, task in enumerate(tasks) if task == name]
        if len(members) <= args.feedback_per_task:
            raise ValueError(f"task {name!r} cannot supply feedback and candidates")
        feedback.extend(rng.sample(members, args.feedback_per_task))
    metadata = {"schema": "orient-metadata-v1", "source_rows_sha256": source_rows_sha256(rows),
                "tasks": tasks, "task_names": names, "feedback_indices": sorted(feedback),
                "reserved_indices": sorted(feedback), "seed": args.seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, metadata)
    try:
        load_metadata(args.out, rows)
    except Exception:
        args.out.unlink()
        raise
    print(f"Prepared {len(feedback)} feedback samples across {len(names)} tasks.")


if __name__ == "__main__":
    main()
