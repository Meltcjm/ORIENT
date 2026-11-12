"""Candidate features f_i,t and causal Critic history b_t (Sections IV and VI)."""
import math
import numpy as np


SIGNALS = ("d_V", "d_T", "delta", "g_xi", "g_theta")
HISTORY_NAMES = ("t_over_H", "previous_reward", "recent_reward_mean",
                 "recent_reward_std", "log1p_t")


def feature_names(task_names):
    return [*SIGNALS, "has_image", "log_prompt_chars", "log_turn_count",
            *(f"task_{task}" for task in task_names),
            *(f"percentile_{name}" for name in SIGNALS)]


def midrank(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("percentiles require a finite vector")
    if not len(values):
        return values
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return ((np.cumsum(counts) - .5 * counts) / len(values))[inverse]


def candidate_features(rows, tasks, task_names, xi_scores, theta_scores):
    from .data import has_image
    if not (len(rows) == len(tasks) == len(xi_scores) == len(theta_scores)):
        raise ValueError("candidate rows, tasks and scores must align")
    lookup = {task: i for i, task in enumerate(task_names)}
    if len(lookup) != len(task_names) or any(task not in lookup for task in tasks):
        raise ValueError("unknown or duplicate task identity")
    raw = np.zeros((len(rows), 5), dtype=np.float32)
    images = np.array([has_image(row) for row in rows], dtype=bool)
    metadata = np.zeros((len(rows), 2), dtype=np.float32)
    for i, (row, xi, theta) in enumerate(zip(rows, xi_scores, theta_scores)):
        if xi.get("invalid_reason") or theta.get("invalid_reason"):
            raise ValueError("candidate has undefined supervised NLL")
        for channel in (("T", "V") if images[i] else ("T",)):
            key = f"ell_{channel}"
            count = f"answer_tokens_{channel}"
            if (any(s.get(key) is None or not np.isfinite(s[key]) or s[key] < 0
                    or type(s.get(count)) is not int or s[count] <= 0 for s in (xi, theta))
                    or xi[count] != theta[count]):
                raise ValueError("reference/current losses or supervised token counts do not align")
        d_T = theta["ell_T"] - xi["ell_T"]
        d_V = theta["ell_V"] - xi["ell_V"] if images[i] else 0.
        g_xi = xi["ell_T"] - xi["ell_V"] if images[i] else 0.
        g_theta = theta["ell_T"] - theta["ell_V"] if images[i] else 0.
        raw[i] = d_V, d_T, d_V - d_T, g_xi, g_theta
        turns = row["conversations"]
        chars = sum(len(t["value"]) for t in turns if t["from"] in ("human", "user"))
        metadata[i] = math.log1p(chars), math.log1p(len(turns))
    ranks = np.full(raw.shape, .5, dtype=np.float32)
    valid = np.zeros(raw.shape, dtype=bool)
    for task in task_names:
        members = np.flatnonzero(np.asarray(tasks) == task)
        for column, name in enumerate(SIGNALS):
            applicable = members if name == "d_T" else members[images[members]]
            values = raw[applicable, column]
            if len(values) and np.ptp(values) > 0:
                ranks[applicable, column] = midrank(values)
                valid[applicable, column] = True
    one_hot = np.eye(len(task_names), dtype=np.float32)[[lookup[t] for t in tasks]]
    X_t = np.column_stack((raw, images, metadata, one_hot, ranks)).astype(np.float32)
    if not np.isfinite(X_t).all():
        raise ValueError("nonfinite candidate features")
    low_costs = ((ranks < .1) & valid).sum(1) / np.maximum(valid.sum(1), 1)
    return X_t, low_costs


def history_state(rewards, H):
    t = len(rewards)
    if type(H) is not int or H <= 0 or not 0 <= t < H or not np.isfinite(rewards).all():
        raise ValueError("history must contain only preceding completed stage rewards")
    recent = np.asarray(rewards[-8:], dtype=np.float64)
    return np.asarray([t / H, rewards[-1] if t else 0.,
                       recent.mean() if t else 0., recent.std() if t else 0.,
                       math.log1p(t)], dtype=np.float32)


def length_percentiles(tasks, answer_counts, candidate_indices):
    result = {}
    for task in dict.fromkeys(tasks[i] for i in candidate_indices):
        members = [i for i in candidate_indices if tasks[i] == task]
        counts = [answer_counts[i] for i in members]
        if any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("length percentiles require positive answer-token counts")
        result.update(zip(members, map(float, midrank(counts))))
    return result


def semantic_clusters(h_t, p, tasks):
    vectors = np.asarray(h_t, dtype=np.float32)
    lengths = np.asarray(p, dtype=np.float32)
    if (vectors.ndim != 2 or not vectors.shape[1] or len(tasks) != len(vectors)
            or lengths.shape != (len(vectors),) or not np.isfinite(vectors).all()
            or not np.isfinite(lengths).all() or np.any((lengths < 0) | (lengths > 1))):
        raise ValueError("clustering requires aligned finite h_i,t and p_i")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("empty pre-answer hidden state")
    vectors = vectors / norms
    labels = np.zeros(len(vectors), dtype=np.int64)
    for task in dict.fromkeys(tasks):
        members = np.flatnonzero(np.asarray(tasks) == task)
        if len(members) < 2:
            continue
        values, percentiles = vectors[members], lengths[members]
        distances = np.maximum(2 * (1 - values @ values.T), 0)
        distances += (percentiles[:, None] - percentiles[None, :]) ** 2
        np.fill_diagonal(distances, np.inf)
        neighbors = distances.argmin(1)
        parents = list(range(len(members)))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for i, j in enumerate(neighbors):
            parents[root(i)] = root(int(j))
        _, labels[members] = np.unique([root(i) for i in range(len(members))], return_inverse=True)
    return labels
