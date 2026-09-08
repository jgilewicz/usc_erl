"""Where does ensemble disagreement concentrate, relative to the training data?

`sigma_ratio_ood < 1` (analyze_ensemble_checkpoints.py) compared disagreement at
uniform random actions against disagreement at the TD3 actor's actions and read
the result as "the ensemble is not epistemic". That comparison is not a test of
epistemic behaviour, because the actor's action is the argmax of the *ensemble
mean* — a point chosen adversarially against the ensemble — while a uniform
action is an average-case point that the buffer (warmup, GA mutants,
exploration noise) covers reasonably well.

This script probes the same checkpoints at points whose relation to the
training distribution is known:

  buffer      (s, a_buf)                training points; sigma floor
  actor       (s, pi_rl(s))             argmax of the ensemble mean
  actor_shuf  (s, pi_rl(s')[perm])      actor's action marginal, off-joint
  corners     (s, a_max * sign(U))      saturated random actions
  population  (s, pi_j(s))              GA members (what the gate scores)
  uniform     (s, U[-a_max, a_max])     the old "OOD" probe
  shuffled    (s, a_buf[perm])          in-marginal, off-joint
  oob x2/x3   (s, c * U)                outside the action bounds
  state x2/x5 (s * c, a_buf)            outside the state support
  ascent/descent  actor action moved by gradient ascent/descent on mean Q

and reports (a) sigma at each, relative to the buffer floor, (b) overestimation
of the ensemble mean against critic_2 at the actor's action, and (c) the
Spearman correlation between sigma and kNN distance to the buffer.

Usage:
    uv run --extra mujoco-envs python scripts/probe_ensemble_uncertainty.py \
        checkpoints/ckpt_Swimmer-v5_seed0_*.pt
"""

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np
import torch
from scipy.stats import spearmanr

from modules.deep_modules import Actor, Critic
from modules.ensemble_module import EnsembleModule

ASCENT_STEPS = 50
ASCENT_LR = 0.02
KNN_SUBSAMPLE = 3000


def _load_actor(ck: dict, state_dict: dict) -> Actor:
    actor = Actor(
        ck["state_dim"],
        ck["action_dim"],
        ck["actor_hidden_dim"],
        ck["action_limit"],
        activation="tanh",
    )
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def _load_critics(ck: dict) -> tuple[EnsembleModule, Critic | None]:
    ens = EnsembleModule(
        ensemble_size=ck["k_ensembles"],
        critic=Critic(ck["state_dim"], ck["action_dim"], activation="elu"),
        rng=np.random.default_rng(0),
    )
    ens.load_state_dict(ck["critic"])
    ens.eval()
    critic_2 = None
    if "critic_2" in ck:
        critic_2 = Critic(ck["state_dim"], ck["action_dim"], activation="elu")
        critic_2.load_state_dict(ck["critic_2"])
        critic_2.eval()
    return ens, critic_2


def _move_along_mean_q(
    ens: EnsembleModule, s: torch.Tensor, a0: torch.Tensor, a_max: float, sign: float
) -> torch.Tensor:
    a = a0.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=ASCENT_LR * a_max)
    for _ in range(ASCENT_STEPS):
        loss = -sign * ens(s, a.clamp(-a_max, a_max))[0].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return a.detach().clamp(-a_max, a_max)


def _knn_distance(
    s_train: torch.Tensor, a_train: torch.Tensor, s: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    x_train = torch.cat([s_train, a_train], dim=1)
    x = torch.cat([s, a], dim=1)
    scale = x_train.std(dim=0, keepdim=True) + 1e-6
    d = torch.cdist(x / scale, x_train / scale)
    return d.min(dim=1).values


def _probe_points(
    ck: dict, ens: EnsembleModule, actor: Actor
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    s = torch.tensor(ck["obs_batch"], dtype=torch.float32)
    a_buf = torch.tensor(ck["act_batch"], dtype=torch.float32)
    a_max = ck["action_limit"]
    with torch.no_grad():
        a_rl = actor(s)
    u = torch.empty_like(a_rl).uniform_(-a_max, a_max)
    perm = torch.randperm(s.shape[0])
    points = {
        "buffer": (s, a_buf),
        "actor": (s, a_rl),
        "actor_shuf": (s, a_rl[perm]),
        "corners": (s, a_max * torch.sign(u)),
        "uniform": (s, u),
        "shuffled": (s, a_buf[perm]),
        "oob_x2": (s, 2.0 * u),
        "oob_x3": (s, 3.0 * u),
        "state_x2": (2.0 * s, a_buf),
        "state_x5": (5.0 * s, a_buf),
        "ascent": (s, _move_along_mean_q(ens, s, a_rl, a_max, +1.0)),
        "descent": (s, _move_along_mean_q(ens, s, a_rl, a_max, -1.0)),
    }
    return points


def analyze(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "act_batch" not in ck:
        raise KeyError(
            f"{ckpt_path} has no 'act_batch'; it predates the checkpoint format "
            "that stores buffer actions — re-run training to regenerate it"
        )
    ens, critic_2 = _load_critics(ck)
    actor = _load_actor(ck, ck["actor"])
    points = _probe_points(ck, ens, actor)
    s, a_buf = points["buffer"]

    sig, mu, abs_a = {}, {}, {}
    with torch.no_grad():
        for name, (ps, pa) in points.items():
            m, sd = ens(ps, pa)
            sig[name], mu[name] = sd.mean().item(), m.mean().item()
            abs_a[name] = (pa.abs().mean() / ck["action_limit"]).item()
        pop = [_load_actor(ck, sd_) for sd_ in ck["population"]]
        pop_sig = np.array([ens(s, p(s))[1].mean().item() for p in pop])
        sig["population"] = float(pop_sig.mean())
        mu["population"] = float(np.mean([ens(s, p(s))[0].mean().item() for p in pop]))
        abs_a["population"] = float(
            np.mean([(p(s).abs().mean() / ck["action_limit"]).item() for p in pop])
        )

    row = {
        "env_id": ck.get("env_id", ""),
        "seed": ck.get("seed", -1),
        "total_steps": ck.get("total_steps", -1),
        "sigma": sig,
        "mu": mu,
        "abs_a": abs_a,
    }
    row.update(_overestimation(ens, critic_2, s, a_buf, points["actor"][1]))
    row.update(_sigma_vs_distance(ens, s, a_buf, points))
    return row


def _overestimation(
    ens: EnsembleModule,
    critic_2: Critic | None,
    s: torch.Tensor,
    a_buf: torch.Tensor,
    a_rl: torch.Tensor,
) -> dict[str, float]:
    if critic_2 is None:
        return {}
    with torch.no_grad():
        gap_actor = (ens(s, a_rl)[0] - critic_2(s, a_rl)).mean().item()
        gap_buf = (ens(s, a_buf)[0] - critic_2(s, a_buf)).mean().item()
        per_member = ens.forward_per_member(s, a_rl).squeeze(-1)
        spread_actor = (per_member.max(0).values - per_member.min(0).values).mean()
    return {
        "mean_minus_q2_actor": gap_actor,
        "mean_minus_q2_buffer": gap_buf,
        "member_range_actor": spread_actor.item(),
    }


def _sigma_vs_distance(
    ens: EnsembleModule,
    s: torch.Tensor,
    a_buf: torch.Tensor,
    points: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    n = min(KNN_SUBSAMPLE, s.shape[0])
    idx = torch.randperm(s.shape[0])[:n]
    ref_mask = torch.ones(s.shape[0], dtype=torch.bool)
    ref_mask[idx] = False
    s_ref, a_ref = s[ref_mask], a_buf[ref_mask]
    sig_all, dist_all = [], []
    with torch.no_grad():
        for name in ("actor", "uniform", "oob_x2", "state_x2"):
            ps, pa = points[name]
            sig_all.append(ens(ps[idx], pa[idx])[1].squeeze(-1))
            dist_all.append(_knn_distance(s_ref, a_ref, ps[idx], pa[idx]))
    sig_v = torch.cat(sig_all).numpy()
    dist_v = torch.cat(dist_all).numpy()
    return {"spearman_sigma_knn": float(spearmanr(sig_v, dist_v).correlation)}


def _print(rows: list[dict]) -> None:
    names = [
        "buffer",
        "population",
        "uniform",
        "shuffled",
        "corners",
        "actor_shuf",
        "actor",
        "oob_x2",
        "oob_x3",
        "state_x2",
        "state_x5",
        "descent",
        "ascent",
    ]
    for r in rows:
        print(f"\n=== {r['env_id']} seed={r['seed']} step={r['total_steps']} ===")
        floor = r["sigma"]["buffer"]
        print(
            f"{'probe':>12} | {'sigma':>9} | {'sigma/buf':>9} | {'mu':>9} | {'|a|/amax':>8}"
        )
        for n in names:
            print(
                f"{n:>12} | {r['sigma'][n]:9.4f} | {r['sigma'][n] / floor:9.3f} | "
                f"{r['mu'][n]:9.3f} | {r['abs_a'][n]:8.3f}"
            )
        print(
            f"old sigma_ratio_ood (uniform/actor) = "
            f"{r['sigma']['uniform'] / r['sigma']['actor']:.3f}"
        )
        if "mean_minus_q2_actor" in r:
            print(
                f"ensemble mean - Q2:  at actor action {r['mean_minus_q2_actor']:+.3f}"
                f"   at buffer action {r['mean_minus_q2_buffer']:+.3f}"
                f"   member max-min at actor action {r['member_range_actor']:.3f}"
            )
        print(
            f"spearman(sigma, kNN distance to buffer) = {r['spearman_sigma_knn']:.3f}"
        )


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    torch.manual_seed(0)
    rows = [analyze(p) for p in paths]
    rows.sort(key=lambda r: (r["env_id"], r["seed"], r["total_steps"]))
    _print(rows)


if __name__ == "__main__":
    main()
