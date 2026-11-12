"""Exact maximum-coverage subset inference (Eqs. 7-10)."""
import numbers

import numpy as np
import torch


def _count(k, n):
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 0 <= k <= n:
        raise ValueError('subset size must be an integer within capacity')
    return int(k)



def _tables(logits, k):
    if logits.ndim != 1 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError('subset logits must be a finite floating-point vector')
    k = _count(k, len(logits))
    # A common offset cancels for fixed k; remove it before the dynamic program.
    z = logits.double() - (logits.double().mean() if len(logits) else 0.)
    zero = z.sum() * 0.
    negative_infinity = zero.new_full((k,), torch.finfo(z.dtype).min)
    logz = torch.cat((zero.reshape(1), negative_infinity))
    expected_score = torch.zeros_like(logz)
    partitions = [logz]
    for count, value in enumerate(z, 1):
        included = torch.cat((logz.new_full((1,), -torch.inf), logz[:-1] + value))
        updated = torch.logaddexp(logz, included)
        include_probability = (included - updated).exp()
        exclude_probability = (logz - updated).exp()
        included_score = torch.cat((zero.reshape(1), expected_score[:-1] + value))
        expected_score = exclude_probability * expected_score + include_probability * included_score
        logz = updated
        partitions.append(logz)
    return z, partitions, logz[k] - expected_score[k]



@torch.no_grad()
def sample_subset(logits, k, generator=None):
    """Exact O(nk) sampling and replayable log probability; no sampled order."""
    z, partitions, entropy = _tables(logits, k)
    need, chosen = int(k), []
    for count in range(len(z), 0, -1):
        if not need:
            break
        if need == count:
            chosen.extend(range(count))
            break
        include = (z[count - 1] + partitions[count - 1][need - 1]
                   - partitions[count][need]).exp().clamp(0., 1.)
        if torch.rand((), device=z.device, generator=generator) < include:
            chosen.append(count - 1)
            need -= 1
    chosen.sort()
    return chosen, float(z[chosen].sum() - partitions[-1][int(k)]), float(entropy)



def _groups(labels, size):
    labels = np.asarray(labels)
    if size == 0 and labels.shape == (0,):
        return []
    if (labels.shape != (size,) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0)):
        raise ValueError('coverage labels must be aligned nonnegative integers')
    return [np.flatnonzero(labels == label).tolist() for label in np.unique(labels)]



def _coefficients(values, degree):
    """Finite log elementary-symmetric coefficients, including degree zero."""
    out = values.sum().mul(0.).reshape(1)
    for value in values:
        take = min(degree + 1, len(out))
        middle = torch.logsumexp(torch.stack((out[1:take], out[:take-1] + value)), 0)
        last = (out[-1:] + value) if len(out) <= degree else out[:0]
        out = torch.cat((out[:1], middle, last))
    return out



def _convolve(left, right, degree):
    count = min(degree + 1, len(left) + len(right) - 1)
    total = torch.arange(count, device=left.device)[:, None]
    r = torch.arange(min(len(right), count), device=left.device)[None, :]
    l = total - r
    valid = (l >= 0) & (l < len(left))
    terms = left[l.clamp(0, len(left)-1)] + right[r]
    return torch.logsumexp(terms.masked_fill(~valid, -torch.inf), 1)



def _coverage_tables(logits, labels, budget):
    if logits.ndim != 1 or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError('coverage logits must be a finite floating-point vector')
    budget = _count(budget, len(logits))
    groups = _groups(labels, len(logits))
    mean = logits.double().mean() if len(logits) else logits.double().sum()
    values = logits.double() - mean
    if budget <= len(groups):
        masses = torch.stack([torch.logsumexp(values[m], 0) for m in groups]) if groups else values
        polynomial = _coefficients(masses, budget)
        return values, groups, masses, None, polynomial[budget] + budget * mean
    extra = budget - len(groups)
    local = [_coefficients(values[m], min(len(m), extra + 1))[1:] for m in groups]
    prefixes = [values.sum().mul(0.).reshape(1)]
    for polynomial in local:
        prefixes.append(_convolve(prefixes[-1], polynomial, extra))
    return values, groups, local, prefixes, prefixes[-1][extra] + budget * mean



def coverage_partition(logits, labels, budget):
    return _coverage_tables(logits, labels, budget)[-1]



def coverage_logp_entropy(logits, labels, chosen):
    chosen = list(chosen)
    if (len(set(chosen)) != len(chosen) or any(isinstance(i, (bool, np.bool_))
            or not isinstance(i, numbers.Integral) or not 0 <= i < len(logits) for i in chosen)):
        raise ValueError('coverage action has repeated or invalid indices')
    groups = _groups(labels, len(logits))
    if len({labels[i] for i in chosen}) != min(len(chosen), len(groups)):
        raise ValueError('action does not achieve maximum feasible cluster coverage')
    differentiate = torch.is_grad_enabled() and logits.requires_grad
    with torch.enable_grad():
        probe = logits.double()
        if not probe.requires_grad:
            probe = probe.detach().requires_grad_(True)
        partition = coverage_partition(probe, labels, len(chosen))
        marginal, = torch.autograd.grad(partition, probe, create_graph=differentiate)
        entropy = partition - (marginal * probe).sum()
        logp = probe[chosen].sum() - partition
    return (logp, entropy) if differentiate else (logp.detach(), entropy.detach())



@torch.no_grad()
def sample_coverage(logits, labels, budget, generator=None, *, compute_entropy=True):
    values, groups, local, prefixes, partition = _coverage_tables(logits, labels, budget)
    chosen = []
    if budget <= len(groups):
        picked, _, _ = sample_subset(local, budget, generator)
        for j in picked:
            position = torch.multinomial(values[groups[j]].softmax(0), 1, generator=generator).item()
            chosen.append(groups[j][position])
    else:
        need = budget - len(groups)
        for j in range(len(groups)-1, -1, -1):
            counts = torch.arange(max(0, need-len(prefixes[j])+1),
                                  min(need+1, len(local[j])), device=values.device)
            mass = local[j][counts] + prefixes[j][need-counts]
            extra = int(counts[torch.multinomial(mass.softmax(0), 1, generator=generator)])
            picked, _, _ = sample_subset(values[groups[j]], extra+1, generator)
            chosen.extend(groups[j][i] for i in picked)
            need -= extra
        if need:
            raise RuntimeError('coverage sampler did not exhaust its budget')
    chosen.sort()
    if not compute_entropy:
        return chosen, float(logits.double()[chosen].sum() - partition), None
    logp, entropy = coverage_logp_entropy(logits, labels, chosen)
    return chosen, float(logp), float(entropy)
