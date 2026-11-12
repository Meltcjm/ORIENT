"""Versioned weights with explicit feature order and normalization."""
from pathlib import Path
import torch

from .features import HISTORY_NAMES
from .policy import OrientActorCritic


def save_checkpoint(path, policy, **metadata):
    value = {**metadata, "schema": "orient-policy-v1", "task_names": list(policy.task_names),
             "feature_names": policy.feature_names, "history_names": list(HISTORY_NAMES),
             "state_dict": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, task_names):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        policy = OrientActorCritic(task_names)
    if (not isinstance(bundle, dict) or bundle.get("schema") != "orient-policy-v1"
            or bundle.get("task_names") != list(task_names)
            or bundle.get("feature_names") != policy.feature_names
            or bundle.get("history_names") != list(HISTORY_NAMES)):
        raise ValueError("policy checkpoint schema or feature/task order differs")
    state = bundle.get("state_dict")
    expected = policy.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("checkpoint tensors do not match ORIENT")
    for key, tensor in state.items():
        if (not torch.is_tensor(tensor) or tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype or not bool(torch.isfinite(tensor).all())
                or (key.endswith("input_scale") and bool((tensor <= 0).any()))):
            raise ValueError(f"invalid checkpoint tensor: {key}")
    policy.load_state_dict(state, strict=True)
    return policy
