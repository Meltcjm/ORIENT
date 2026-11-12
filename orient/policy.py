"""Actor, Critic, proportional task quotas, and exact stage distributions."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .features import feature_names
from .subset import _count, _groups, coverage_partition, sample_coverage


@dataclass(frozen=True)
class TaskConstraint:
    task: str
    indices: list[int]
    clusters: list[int]
    k: int


def task_constraints(tasks, clusters, K_t, task_names, t=0):
    K_t = _count(K_t, len(tasks))
    _groups(clusters, len(tasks))
    if len(set(task_names)) != len(task_names) or any(task not in task_names for task in tasks):
        raise ValueError("unknown or repeated task name")
    members = [[i for i, task in enumerate(tasks) if task == name] for name in task_names]
    capacities = [len(m) for m in members]
    N = len(tasks)
    quotas = [K_t * count // max(N, 1) for count in capacities]
    # Rotate only exact remainder ties; preserve proportional quotas.
    order = list(range(len(task_names)))
    rotation = t % max(len(order), 1)
    order = order[rotation:] + order[:rotation]
    priority = {index: rank for rank, index in enumerate(order)}
    remaining = K_t - sum(quotas)
    for j in sorted(order, key=lambda j: (-(K_t * capacities[j] % max(N, 1)), priority[j]))[:remaining]:
        quotas[j] += 1
    return tuple(TaskConstraint(task, indices, [int(clusters[i]) for i in indices], k)
                 for task, indices, k in zip(task_names, members, quotas))


def validate_constraints(constraints, size):
    indices = []
    for cell in constraints:
        _count(cell.k, len(cell.indices))
        _groups(cell.clusters, len(cell.indices))
        indices.extend(cell.indices)
    if (any(type(i) is not int for i in indices) or sorted(indices) != list(range(size))
            or len({cell.task for cell in constraints}) != len(constraints)):
        raise ValueError("task constraints must partition all candidates")


def log_partition(z, constraints):
    validate_constraints(constraints, len(z))
    return z.double().sum() * 0. + sum(
        coverage_partition(z[cell.indices], cell.clusters, cell.k) for cell in constraints)


def distribution_moments(z, constraints, create_graph=False):
    with torch.enable_grad():
        probe = z.double()
        if not probe.requires_grad:
            probe = probe.detach().requires_grad_(True)
        log_Z = log_partition(probe, constraints)
        mu, = torch.autograd.grad(log_Z, probe, create_graph=create_graph)
    return (log_Z, mu) if create_graph else (log_Z.detach(), mu.detach())


def validate_selection(S_t, constraints, size):
    validate_constraints(constraints, size)
    if (any(type(i) is not int or not 0 <= i < size for i in S_t)
            or len(set(S_t)) != len(S_t)):
        raise ValueError("invalid selected subset")
    selected = set(S_t)
    for cell in constraints:
        local = [j for j, i in enumerate(cell.indices) if i in selected]
        if (len(local) != cell.k
                or len({cell.clusters[j] for j in local}) != min(cell.k, len(set(cell.clusters)))):
            raise ValueError("selection violates task quota or maximum cluster coverage")


@torch.no_grad()
def sample_action(z, constraints, generator):
    validate_constraints(constraints, len(z))
    S_t = []
    for cell in constraints:
        chosen, _, _ = sample_coverage(z[cell.indices], cell.clusters, cell.k, generator,
                                       compute_entropy=False)
        S_t.extend(cell.indices[j] for j in chosen)
    S_t.sort()
    validate_selection(S_t, constraints, len(z))
    log_pi = z.double()[S_t].sum() - log_partition(z, constraints)
    return S_t, float(log_pi)


class OrientActorCritic(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        self.task_names = tuple(task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task names must be nonempty and unique")
        self.feature_names = feature_names(self.task_names)
        width = len(self.feature_names)
        self.actor = nn.Sequential(nn.Linear(width, 256), nn.GELU(),
                                   nn.Linear(256, 128), nn.GELU(),
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.actor.register_buffer("input_mean", torch.zeros(width))
        self.actor.register_buffer("input_scale", torch.ones(width))
        self.critic = nn.Sequential(nn.Linear(5, 32), nn.Tanh(),
                                    nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.register_buffer("critic_input_mean", torch.zeros(5))
        self.register_buffer("critic_input_scale", torch.ones(5))
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=.5)
                nn.init.zeros_(layer.bias)
        for network in (self.actor, self.critic):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def scores(self, X_t):
        if (X_t.ndim != 2 or X_t.shape[1] != len(self.feature_names)
                or not bool(torch.isfinite(X_t).all())):
            raise ValueError("Actor inputs do not match the feature schema")
        z = self.actor((X_t.float() - self.actor.input_mean) / self.actor.input_scale).squeeze(-1)
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("nonfinite Actor scores")
        return z

    def value(self, b_t):
        if b_t.shape[-1] != 5 or not bool(torch.isfinite(b_t).all()):
            raise ValueError("Critic requires finite five-dimensional causal history")
        value = self.critic((b_t.float() - self.critic_input_mean) / self.critic_input_scale).squeeze(-1)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite Critic value")
        return value


def feedback_loss(scores, tasks, task_names):
    if len(scores) != len(tasks) or set(tasks) != set(task_names):
        raise ValueError("feedback scores must cover every task")
    values = np.asarray([score.get("ell_V" if score.get("has_image") else "ell_T", np.nan)
                         for score in scores], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("feedback contains undefined or negative NLL")
    means = {task: float(values[np.asarray(tasks) == task].mean()) for task in task_names}
    return float(np.mean(list(means.values()))), means
