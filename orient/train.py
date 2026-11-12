"""Run Algorithm 1 through one continuous native LLaVA training epoch."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys

import torch

from .cache import load_reference_cache, scoring_contract
from .checkpoint import load_checkpoint, save_checkpoint
from .data import feedback_groups, file_sha256, load_metadata, load_rows, plan_stages, write_json
from .distributed import Collective
from .features import candidate_features, history_state, length_percentiles, semantic_clusters
from .policy import OrientActorCritic, feedback_loss, sample_action, task_constraints
from .ppo import MAX_ACTOR_STEPS, PPOConfig, Transition, make_optimizer, update_policy
from .acceleration import ScoringRuntime
from .prefetch import OnlineInputPrefetch
from .scoring_inputs import ScoringInputCache


def load_inputs(args):
    rows = load_rows(args.data_path)
    metadata = load_metadata(args.metadata, rows)
    xi, contract = load_reference_cache(args.reference_cache, rows, image_folder=args.image_folder,
                                        return_contract=True)
    feedback = metadata["feedback_indices"]
    if any(xi[i].get("invalid_reason") for i in feedback):
        raise ValueError("fixed feedback includes rows without supervised answer tokens")
    reserved = set(metadata["reserved_indices"])
    candidates = [i for i, score in enumerate(xi) if i not in reserved and not score.get("invalid_reason")]
    stages = plan_stages(candidates, args.rho, args.B, args.n, args.seed)
    return rows, metadata, xi, candidates, stages, contract


class Controller:
    def __init__(self, args, rows, metadata, xi, candidates, stages, data_args, collective):
        self.args, self.rows, self.metadata, self.xi = args, rows, metadata, xi
        self.stages, self.data_args, self.collective = stages, data_args, collective
        self.tasks, self.task_names = metadata["tasks"], metadata["task_names"]
        counts = [s["answer_tokens_V" if s["has_image"] else "answer_tokens_T"] for s in xi]
        self.p = length_percentiles(self.tasks, counts, candidates)
        self.rewards, self.records, self.pending = [], [], None
        self.active, self.finished, self.L_F = None, False, None
        self.policy, self.optimizer = None, None
        self.config = PPOConfig(**{name: getattr(args, name) for name in PPOConfig.__dataclass_fields__})
        self.generator = torch.Generator().manual_seed(args.seed)
        self.trainer = None
        self.feedback_cache = ScoringInputCache(1024**3)
        self.runtime = ScoringRuntime(answer_sliced=args.answer_sliced_logits)
        self.prefetch = None

    def initialize_policy(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.args.seed)
            self.policy = (load_checkpoint(self.args.policy_init, self.task_names) if self.args.policy_init
                           else OrientActorCritic(self.task_names))
        self.optimizer = make_optimizer(self.policy, self.config)

    def candidate_cache(self, stage, trainer):
        if self.prefetch is None:
            # Forking DataLoader workers while preprocessing runs can deadlock.
            self.prefetch = OnlineInputPrefetch(trainer.args.dataloader_num_workers == 0)
        cache = self.prefetch.take(stage.t)
        if self.prefetch.enabled and stage.t + 1 < len(self.stages):
            next_stage = self.stages[stage.t + 1]
            local_indices = next_stage.P_t[self.collective.rank::self.collective.world_size]
            self.prefetch.submit(next_stage.t, [self.rows[i] for i in local_indices],
                                 trainer.tokenizer, self.data_args, str(self.args.image_folder))
        return cache if cache is not None else ScoringInputCache(512 * 1024**2)

    def close(self):
        if self.prefetch is not None:
            self.prefetch.close()

    def score(self, indices, trainer, capture_hidden=False, stage=None):
        from .scoring import score_rows

        def local(selected):
            cache = self.candidate_cache(stage, trainer) if capture_hidden else self.feedback_cache
            with trainer.accelerator.autocast():
                return score_rows(trainer.model, trainer.tokenizer, [self.rows[i] for i in selected],
                                  str(self.args.image_folder), self.data_args, self.args.score_batch_size,
                                  input_cache=cache, capture_hidden=capture_hidden, runtime=self.runtime,
                                  feedback_only=not capture_hidden)

        return self.collective.score(indices, local)

    def evaluate_feedback(self, trainer):
        indices, scores = [], []
        for group in feedback_groups(self.metadata):
            values = self.score(group, trainer)
            if self.collective.rank == 0:
                indices.extend(group)
                scores.extend(values)
        return self.collective.main(lambda: feedback_loss(
            scores, [self.tasks[i] for i in indices], self.task_names))

    def record_training_configuration(self, trainer):
        from .native import training_configuration

        def record():
            path = self.args.out / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["native"] = training_configuration(trainer.args)
            manifest["native"]["optimizer_updates"] = trainer.state.max_steps
            manifest["native"]["warmup_updates"] = trainer.args.get_warmup_steps(trainer.state.max_steps)
            write_json(path, manifest)

        self.collective.main(record)

    def begin_stage(self, stage, trainer):
        if self.active is not None or self.finished:
            raise RuntimeError("stage began before the preceding stage completed")
        if self.L_F is None:
            self.L_F, _ = self.evaluate_feedback(trainer)
        theta = self.score(stage.P_t, trainer, capture_hidden=True, stage=stage)

        def select():
            tasks = [self.tasks[i] for i in stage.P_t]
            X_t, low_costs = candidate_features(
                [self.rows[i] for i in stage.P_t], tasks, self.task_names,
                [self.xi[i] for i in stage.P_t], theta)
            clusters = semantic_clusters([s["h"] for s in theta], [self.p[i] for i in stage.P_t], tasks)
            constraints = task_constraints(tasks, clusters, stage.K_t, self.task_names, stage.t)
            X_t = torch.from_numpy(X_t)
            b_t = torch.from_numpy(history_state(self.rewards, len(self.stages)))
            if self.pending is not None:
                self.complete_transition(b_t)
            with torch.no_grad():
                z = self.policy.scores(X_t)
                S_t, log_pi = sample_action(z, constraints, self.generator)
                V = float(self.policy.value(b_t))
            self.pending = Transition(X_t, b_t, constraints, S_t, z.detach().clone(), log_pi, V,
                                      torch.as_tensor(low_costs, dtype=torch.float64))
            return [stage.P_t[i] for i in S_t]

        selected = self.collective.main(select)
        trainer.train_dataset.assign(stage, selected)
        self.active = {"stage": stage, "global_step_before": trainer.state.global_step,
                       "selected": selected}

    def end_stage(self, stage, trainer):
        if self.active is None or self.active["stage"] is not stage:
            raise RuntimeError("wrong active stage")
        L_after, per_task = self.evaluate_feedback(trainer)
        r_t = self.L_F - L_after
        record = {"t": stage.t, "K_t": stage.K_t, "candidate_count": len(stage.P_t),
                  "selected_indices": self.active["selected"],
                  "global_step": int(trainer.state.global_step), "L_F_before": self.L_F,
                  "L_F_after": L_after, "feedback_by_task": per_task, "r_t": r_t}

        def save():
            self.pending.r_t = r_t
            self.rewards.append(r_t)
            self.records.append(record)
            write_json(self.args.out / "stages" / f"{stage.t:05d}.json", record)

        self.collective.main(save)
        self.L_F, self.active = L_after, None

    def complete_transition(self, b_next):
        if self.pending is None:
            raise RuntimeError("no transition to update")
        metrics = update_policy(self.policy, self.optimizer, self.pending, b_next, self.config)
        record = self.records[-1]
        record["policy_update"] = metrics
        write_json(self.args.out / "stages" / f"{record['t']:05d}.json", record)
        self.pending = None

    def finish(self, trainer):
        def save():
            self.complete_transition(None)
            save_checkpoint(self.args.out / "policy.pt", self.policy,
                            completed_stages=len(self.records), seed=self.args.seed)
            write_json(self.args.out / "selected_indices.json", trainer.train_dataset.assignment)
            write_json(self.args.out / "summary.json", {
                "completed_stages": len(self.records), "selected_count": len(trainer.train_dataset),
                "optimizer_updates": trainer.state.global_step, "L_F_final": self.L_F,
                "sum_reward": sum(self.rewards)})
        self.collective.main(save)
        self.finished = True


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("data-path", "metadata", "reference-cache", "reference-model", "model-name-or-path",
                 "vision-tower", "image-folder", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--policy-init", type=Path)
    parser.add_argument("--rho", type=float, default=.2)
    parser.add_argument("--B", type=int, required=True, help="effective global SFT batch size")
    parser.add_argument("--n", type=int, required=True, help="SFT optimizer updates per full stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--answer-sliced-logits", action="store_true",
                        help="project only answer positions during scoring; native heads are the default")
    parser.add_argument("--attn-implementation", choices=("eager", "flash_attention_2"), default="flash_attention_2")
    parser.add_argument("--dry-run", action="store_true", help="validate data and budgets without loading models")
    for name, field in PPOConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=field.default)
    return parser


def implementation_identity():
    packages = {}
    for name in ("torch", "numpy", "transformers", "peft", "accelerate", "Pillow", "deepspeed", "flash-attn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages,
            "orient_sources": {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
            "llava_sources": {name: file_sha256(module.__file__) for name, module in sorted(sys.modules.items())
                              if (name == "llava" or name.startswith("llava."))
                              and (getattr(module, "__file__", None) or "").endswith(".py")}}


def main(argv=None):
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.out.exists():
        parser.error("output already exists; use a fresh directory")
    if args.score_batch_size <= 0:
        parser.error("score-batch-size must be positive")
    PPOConfig(**{key: getattr(args, key) for key in PPOConfig.__dataclass_fields__})
    rows, metadata, xi, candidates, stages, cached_contract = load_inputs(args)
    if args.policy_init:
        load_checkpoint(args.policy_init, metadata["task_names"])
    report = {"N": len(candidates), "K": sum(s.K_t for s in stages), "H": len(stages),
              "rho": args.rho, "B": args.B, "n": args.n, "source_rows": len(rows),
              "feedback_rows": len(metadata["feedback_indices"]),
              "first_stage_candidates": len(stages[0].P_t),
              "last_stage_candidates": len(stages[-1].P_t)}
    if args.dry_run:
        native = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        native.add_argument("--per_device_train_batch_size", type=int, default=8)
        native.add_argument("--gradient_accumulation_steps", type=int, default=4)
        batch, _ = native.parse_known_args(extra)
        effective = batch.per_device_train_batch_size * batch.gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE", "1"))
        if effective != args.B:
            parser.error(f"B={args.B} differs from native global batch {effective}")
        print(json.dumps({**report, "dry_run": True, "native_model_checked": False}, indent=2))
        return
    from .native import (OnlineSelectedDataset, build_model, make_stage_callback,
                         make_trainer_class, parse_native_arguments, save_native_model, training_configuration)
    from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
    from transformers import set_seed

    model_args, data_args, training_args = parse_native_arguments(args, extra)
    with ExitStack() as cleanup:
        collective = cleanup.enter_context(Collective())
        effective = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        if effective != args.B:
            raise ValueError(f"B={args.B} differs from native global batch {effective}")
        contract = collective.main(lambda: scoring_contract(model_args, data_args, training_args, args.reference_model))
        if cached_contract != contract:
            raise ValueError("reference model or preprocessing differs from cache")

        def initialize_output():
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "stages").mkdir()
            manifest = {"schema": "orient-run-v1", **report, "seed": args.seed,
                        "source_rows_sha256": metadata["source_rows_sha256"],
                        "inputs": {key: {"path": str(getattr(args, key).resolve()),
                                         "sha256": file_sha256(getattr(args, key))}
                                   for key in ("data_path", "metadata", "reference_cache")},
                        "scoring_contract": contract,
                        "policy_init_sha256": file_sha256(args.policy_init) if args.policy_init else None,
                        "ppo": {**asdict(PPOConfig(**{k: getattr(args, k) for k in PPOConfig.__dataclass_fields__})),
                                "max_actor_steps": MAX_ACTOR_STEPS, "critic_steps": 1,
                                "update_every_stages": 1},
                        "native": training_configuration(training_args),
                        "implementation": implementation_identity(),
                        "score_batch_size": args.score_batch_size,
                        "feedback_scoring_groups": feedback_groups(metadata),
                        "freeze_backbone": model_args.freeze_backbone,
                        "attn_implementation": args.attn_implementation,
                        "scoring_acceleration": {
                            "answer_sliced_logits": args.answer_sliced_logits,
                            "input_prefetch": training_args.dataloader_num_workers == 0,
                            "dataloader_num_workers": training_args.dataloader_num_workers,
                            "feedback_input_cache_bytes": 1024**3,
                            "prefetch_cache_bytes": 4 * 1024**3,
                            "frozen_vision_cache_bytes": 512 * 1024**2},
                        "world_size": collective.world_size}
            write_json(args.out / "manifest.json", manifest)

        collective.main(initialize_output)
        set_seed(args.seed)
        model, tokenizer = build_model(model_args, data_args, training_args, args.attn_implementation)
        source = LazySupervisedDataset(str(args.data_path), tokenizer, data_args)
        dataset = OnlineSelectedDataset(source, stages)
        controller = Controller(args, rows, metadata, xi, candidates, stages, data_args, collective)
        cleanup.callback(controller.close)
        collective.main(controller.initialize_policy)
        trainer = make_trainer_class()(model=model, tokenizer=tokenizer, args=training_args,
                                      train_dataset=dataset, data_collator=DataCollatorForSupervisedDataset(tokenizer),
                                      callbacks=[make_stage_callback(controller)])
        controller.trainer, trainer.orient_controller = trainer, controller
        trainer.train()

        def export():
            trainer.save_state()
            save_native_model(trainer)

        collective.main(export)
        collective.main(lambda: print(f"ORIENT completed {len(stages)} stages; output: {args.out}"))


if __name__ == "__main__":
    main()
