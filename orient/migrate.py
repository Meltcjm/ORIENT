"""Convert only the legacy actor24/history5 proportional-coverage policy."""
import argparse
from pathlib import Path
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import file_sha256
from .policy import OrientActorCritic
from .prepare import LLAVA_TASK_NAMES


def convert_bundle(bundle):
    names = list(LLAVA_TASK_NAMES)
    raw = ["rv", "rt", "delta_g", "g_xi", "g_theta"]
    actor_names = raw + ["image", "log_prompt_chars", "log_turn_count"] + [f"task_{t}" for t in names] + [f"task_rank_{s}" for s in raw]
    history = ["progress", "last_anchor_reward", "past8_anchor_reward_mean", "past8_anchor_reward_std", "log1p_stage"]
    if (not isinstance(bundle, dict) or bundle.get("schema") != "caps-online-anchor-actor24-history5-v4"
            or bundle.get("selection_mode") != "task11-coverage-v4"
            or bundle.get("task_quota_rule") != "capacity"
            or bundle.get("reward_aggregation") != "task11_macro_mean"
            or bundle.get("actor_feature_names") != actor_names
            or bundle.get("critic_state_names") != history
            or bundle.get("critic_state_schema") != "history5"
            or bundle.get("critic_state_dim") != 5 or bundle.get("task_names") != names
            or "structure_calibration" in bundle):
        raise ValueError("only actor24/history5, capacity quotas and maximum coverage can be migrated")
    architecture = bundle.get("architecture", {})
    for name, dimension, widths, activation in (("actor", 24, [256, 128, 64], "GELU"),
                                                ("critic", 5, [32, 32], "Tanh")):
        spec = architecture.get(name, {})
        if (spec.get("input_dim") != dimension or spec.get("widths") != widths
                or spec.get("activation") != activation or spec.get("output_dim") != 1):
            raise ValueError("unsupported legacy architecture")
    state = {f"actor.{key}": value.clone() for key, value in bundle["actor"].items()}
    state.update({f"critic.{key}": value.clone() for key, value in bundle["critic"].items()})
    for key in ("critic_input_mean", "critic_input_scale"):
        state[key] = bundle[key].clone()
    if bundle.get("critic_mode") == "popart" and bundle.get("value_normalization") is True:
        scale, mean = bundle.get("value_scale"), bundle.get("value_mean")
        if (any(not torch.is_tensor(v) or v.shape != () or not bool(torch.isfinite(v)) for v in (scale, mean))
                or float(scale) <= 0):
            raise ValueError("invalid legacy value coordinates")
        state["critic.4.weight"].mul_(scale)
        state["critic.4.bias"].mul_(scale).add_(mean)
    elif bundle.get("critic_mode") != "raw" or bundle.get("value_normalization") is not False:
        raise ValueError("unsupported legacy critic output coordinates")
    policy = OrientActorCritic(names)
    expected = policy.state_dict()
    if set(state) != set(expected):
        raise ValueError("legacy state contains missing or extra tensors")
    for key, value in state.items():
        if (value.shape != expected[key].shape or value.dtype != expected[key].dtype
                or not bool(torch.isfinite(value).all())
                or (key.endswith("input_scale") and bool((value <= 0).any()))):
            raise ValueError(f"invalid legacy tensor {key}")
    policy.load_state_dict(state, strict=True)
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists")
    bundle = torch.load(args.policy, map_location="cpu", weights_only=True)
    policy = convert_bundle(bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, policy, migrated_from_sha256=file_sha256(args.policy),
                    migration="weights and fixed input normalization; no optimizer or run resume")
    load_checkpoint(args.out, policy.task_names)
    print(f"Converted policy: {args.out}")


if __name__ == "__main__":
    main()
