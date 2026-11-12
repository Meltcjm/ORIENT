"""Per transition: at most two Actor steps, one Critic step, and exact KL control."""
from copy import deepcopy
from dataclasses import dataclass
import math
import torch

from .policy import distribution_moments, log_partition, validate_selection


MAX_ACTOR_STEPS = 2


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = .95
    epsilon: float = .2
    beta_critic: float = .5
    beta_ent: float = .01
    kl_limit: float = .03
    max_grad_norm: float = .5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = .01

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values())
                or not 0 <= self.gamma <= 1 or not 0 < self.epsilon < 1
                or min(self.beta_critic, self.beta_ent, self.weight_decay) < 0
                or min(self.kl_limit, self.max_grad_norm, self.actor_lr, self.critic_lr) <= 0):
            raise ValueError("invalid PPO hyperparameters")


@dataclass
class Transition:
    X_t: torch.Tensor
    b_t: torch.Tensor
    constraints: tuple
    S_t: list[int]
    z_old: torch.Tensor
    log_pi_old: float
    V_old: float
    low_costs: torch.Tensor
    r_t: float = 0.


def make_optimizer(policy, config):
    return torch.optim.AdamW([
        {"params": policy.actor.parameters(), "lr": config.actor_lr},
        {"params": policy.critic.parameters(), "lr": config.critic_lr},
    ], weight_decay=config.weight_decay)


def update_policy(policy, optimizer, transition, b_next, config):
    tr = transition
    K_t = len(tr.S_t)
    validate_selection(tr.S_t, tr.constraints, len(tr.X_t))
    if (not K_t or tr.low_costs.shape != (len(tr.X_t),)
            or not bool(torch.isfinite(tr.low_costs).all())
            or bool(((tr.low_costs < 0) | (tr.low_costs > 1)).any())
            or not all(math.isfinite(x) for x in (tr.r_t, tr.V_old, tr.log_pi_old))):
        raise ValueError("invalid transition or fixed low-range costs")
    with torch.no_grad():
        next_value = 0. if b_next is None else float(policy.value(b_next))
        R_hat = tr.r_t + config.gamma * next_value
        Adv = R_hat - tr.V_old
        replay = policy.scores(tr.X_t)
        if not torch.allclose(replay, tr.z_old, rtol=1e-6, atol=1e-7):
            raise ValueError("Actor changed between sampling and its transition update")
    log_Z_old, mu_old = distribution_moments(tr.z_old, tr.constraints)
    replay_log_pi = tr.z_old.double()[tr.S_t].sum() - log_Z_old
    if abs(float(replay_log_pi) - tr.log_pi_old) > 1e-6:
        raise ValueError("cached action log probability does not replay")
    L_critic = (policy.value(tr.b_t) - R_hat).square().mean()
    actor, critic = list(policy.actor.parameters()), list(policy.critic.parameters())
    steps = []
    for step in range(MAX_ACTOR_STEPS):
        optimizer.zero_grad(set_to_none=True)
        z = policy.scores(tr.X_t)
        log_Z, mu = distribution_moments(z, tr.constraints, create_graph=True)
        log_pi = z.double()[tr.S_t].sum() - log_Z
        omega = torch.exp(log_pi - tr.log_pi_old)
        L_actor = -torch.minimum(omega * Adv, omega.clamp(1 - config.epsilon, 1 + config.epsilon) * Adv)
        Ent = (log_Z - (mu * z.double()).sum()) / K_t
        L_low = (mu * tr.low_costs.detach()).sum() / K_t
        loss = L_actor - config.beta_ent * Ent + L_low
        if step == 0:
            loss = loss + config.beta_critic * L_critic
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite policy loss")
        loss.backward()
        for parameters in (actor, critic) if step == 0 else (actor,):
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
        before_step = [p.detach().clone() for p in actor]
        before_state = {p: deepcopy(optimizer.state.get(p)) for p in actor}
        optimizer.step()
        # Keep the log field compatible: a whole step is either accepted or reverted.
        accepted_fraction = 0.
        with torch.no_grad():
            try:
                candidate = policy.scores(tr.X_t)
                partition = log_partition(candidate, tr.constraints)
                # Both steps measure cumulative KL from the original sampling policy.
                KL = partition - log_Z_old - (mu_old * (candidate.double() - tr.z_old.double())).sum()
                omega_after = (candidate.double()[tr.S_t].sum() - partition - tr.log_pi_old).exp()
                if (bool(torch.isfinite(KL)) and -1e-7 <= float(KL) <= config.kl_limit
                        and bool(torch.isfinite(omega_after))):
                    accepted_fraction = 1.
            except FloatingPointError:
                pass
            if not accepted_fraction:
                # Restore only this Actor step, preserving earlier accepted steps
                # and the Critic update. Restore AdamW moments and its step counter too.
                for parameter, before in zip(actor, before_step):
                    parameter.copy_(before)
                    if before_state[parameter] is None:
                        optimizer.state.pop(parameter, None)
                    else:
                        optimizer.state[parameter] = before_state[parameter]
                candidate = policy.scores(tr.X_t)
                partition = log_partition(candidate, tr.constraints)
                KL = partition - log_Z_old - (mu_old * (candidate.double() - tr.z_old.double())).sum()
                omega_after = (candidate.double()[tr.S_t].sum() - partition - tr.log_pi_old).exp()
        if (not bool(torch.isfinite(KL)) or float(KL) < -1e-7 or float(KL) > config.kl_limit
                or not bool(torch.isfinite(omega_after))):
            raise FloatingPointError("final policy failed exact KL or probability-ratio validation")
        ratio = float(omega.detach())
        steps.append({"step": step + 1, "omega_before": ratio, "omega_after": float(omega_after),
                      "clip_active": (Adv > 0 and ratio > 1 + config.epsilon
                                      or Adv < 0 and ratio < 1 - config.epsilon),
                      "L_actor": float(L_actor.detach()), "L_low": float(L_low.detach()),
                      "Ent": float(Ent.detach()), "loss": float(loss.detach()),
                      "exact_KL": max(0., float(KL)), "accepted_fraction": accepted_fraction})
        if not accepted_fraction or float(KL) >= config.kl_limit:
            break
    first, last = steps[0], steps[-1]
    return {"r_t": tr.r_t, "R_hat": R_hat, "Adv": Adv,
            "L_actor": first["L_actor"], "L_critic": float(L_critic.detach()),
            "L_low": first["L_low"], "Ent": first["Ent"], "loss": first["loss"],
            "exact_KL": last["exact_KL"], "omega_after": last["omega_after"],
            "accepted_fraction": last["accepted_fraction"], "actor_steps": steps,
            "actor_optimizer_steps": sum(s["accepted_fraction"] > 0 for s in steps),
            "critic_optimizer_steps": 1}
