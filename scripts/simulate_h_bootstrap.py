"""Offline simulation of the H-bootstrap fitness estimator (see H_bootstrap.md).

Rolls out every population member of an ensemble checkpoint once, caches per-step
rewards and per-member Q along the trajectory, then evaluates any rollout length
H without touching the environment again. Compares budget-allocation strategies
at equal env-step budgets against the real return.

Usage:
    uv run --extra mujoco-envs python scripts/simulate_h_bootstrap.py \
        checkpoints/ckpt_Hopper-v5_seed0_100000.pt checkpoints/ckpt_Swimmer-v5_seed0_100000.pt

    uv run --extra mujoco-envs python scripts/simulate_h_bootstrap.py \
        --verify checkpoints/ckpt_Hopper-v5_seed0_100000.pt
"""

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scipy.stats import norm, spearmanr

from common.h_bootstrap import TailCalibrator
from entry_point import make_env
from modules.deep_modules import Actor, Critic
from modules.ensemble_module import EnsembleModule

GAMMA = 0.99
T_MAX = 1000
CHUNK = 25
H_MIN = 25
ELITE_K = 10
BUDGETS = (0.05, 0.10, 0.25, 0.50)
STRATEGIES = ("uniform", "random_gate", "sigma_greedy", "adaptive")
SWEEP_H = (0, 5, 10, 25, 50, 100, 200, 500, 1000)
CACHE_DIR = REPO_ROOT / "outputs" / "h_bootstrap_cache"


@dataclass
class Trajectory:
    rewards: np.ndarray  # (L,)
    q: np.ndarray  # (K, L) per-member Q(s_t, pi(s_t))
    terminated: bool

    @property
    def length(self) -> int:
        return len(self.rewards)

    @property
    def real(self) -> float:
        return float(self.rewards.sum())

    @property
    def real_disc(self) -> float:
        return float((GAMMA ** np.arange(self.length) * self.rewards).sum())

    def estimate(self, h: int, tail: str) -> tuple[float, float]:
        h = min(h, self.length)
        if h >= self.length:
            return (self.real_disc if tail == "disc" else self.real), 0.0
        q_h = self.q[:, h]
        if tail == "lin":
            # assumes survival to T_MAX: biased upward on terminating envs
            per_member = self.rewards[:h].sum() + (T_MAX - h) * (1.0 - GAMMA) * q_h
        else:
            disc = (GAMMA ** np.arange(h) * self.rewards[:h]).sum()
            per_member = disc + GAMMA**h * q_h
        return float(per_member.mean()), float(per_member.std(ddof=1))


def _load_actor(ck: dict, sd: dict) -> Actor:
    actor = Actor(
        ck["state_dim"],
        ck["action_dim"],
        ck["actor_hidden_dim"],
        ck["action_limit"],
        activation="tanh",
    )
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def _rollout(policy: Actor, env, ens: EnsembleModule) -> Trajectory:
    obs, _ = env.reset(seed=0)
    states, rewards, terminated = [], [], False
    done = False
    while not done:
        states.append(obs)
        with torch.no_grad():
            a = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        a = np.clip(a.squeeze(0).numpy(), env.action_space.low, env.action_space.high)
        obs, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        terminated, done = bool(term), term or trunc
    s = torch.tensor(np.array(states), dtype=torch.float32)
    with torch.no_grad():
        q = ens.forward_per_member(s, policy(s)).squeeze(-1).numpy()
    return Trajectory(np.array(rewards), q, terminated)


def collect(ckpt_path: str) -> tuple[dict, list[Trajectory]]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cache = CACHE_DIR / f"{Path(ckpt_path).stem}_trajs.pt"
    if cache.exists():
        raw = torch.load(cache, map_location="cpu", weights_only=False)
        return ck, [Trajectory(*t) for t in raw]

    ens = EnsembleModule(
        ck["k_ensembles"],
        Critic(ck["state_dim"], ck["action_dim"], activation="elu"),
        np.random.default_rng(0),
    )
    ens.load_state_dict(ck["critic"])
    ens.eval()
    env = make_env(ck["env_id"])
    trajs = [_rollout(_load_actor(ck, sd), env, ens) for sd in ck["population"]]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save([(t.rewards, t.q, t.terminated) for t in trajs], cache)
    return ck, trajs


def _allocate(
    trajs: list[Trajectory],
    budget: int,
    strategy: str,
    tail: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(trajs)
    lengths = np.array([t.length for t in trajs])
    if strategy == "uniform":
        return np.minimum(np.full(n, budget // n), lengths)

    h = np.minimum(np.full(n, min(H_MIN, budget // n)), lengths)
    spent = int(h.sum())
    if strategy == "random_gate":
        for j in rng.permutation(n):
            extra = lengths[j] - h[j]
            if spent + extra > budget:
                continue
            h[j], spent = lengths[j], spent + extra
        return h

    while spent < budget:
        est = np.array([trajs[j].estimate(int(h[j]), tail) for j in range(n)])
        f_hat, sigma = est[:, 0], est[:, 1]
        open_mask = (h < lengths) & (sigma > 0)
        if not open_mask.any():
            break
        if strategy == "sigma_greedy":
            score = np.where(open_mask, sigma, -np.inf)
        else:
            f_cut = np.sort(f_hat)[-ELITE_K]
            p = norm.cdf(-np.abs(f_hat - f_cut) / np.maximum(sigma, 1e-8))
            score = np.where(open_mask, p, -np.inf)
        j = int(np.argmax(score))
        step = int(min(CHUNK, lengths[j] - h[j], budget - spent))
        h[j], spent = h[j] + step, spent + step
    return h


def _metrics(f_hat: np.ndarray, real: np.ndarray) -> dict[str, float]:
    top_est = set(np.argsort(-f_hat)[:ELITE_K])
    top_real = set(np.argsort(-real)[:ELITE_K])
    regret = real[list(top_real)].mean() - real[list(top_est)].mean()
    return {
        "spearman": float(spearmanr(f_hat, real).correlation),
        "top10_prec": len(top_est & top_real) / ELITE_K,
        "regret": float(regret),
    }


def simulate(trajs: list[Trajectory], tail: str, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    real = np.array([t.real for t in trajs])
    full_cost = int(sum(t.length for t in trajs))
    rows = []
    f0 = np.array([t.estimate(0, tail)[0] for t in trajs])
    rows.append({"budget": 0.0, "strategy": "H=0 (s_0 only)", **_metrics(f0, real)})
    for b in BUDGETS:
        budget = int(b * full_cost)
        for strategy in STRATEGIES:
            h = _allocate(trajs, budget, strategy, tail, rng)
            f_hat = np.array(
                [t.estimate(int(h[j]), tail)[0] for j, t in enumerate(trajs)]
            )
            rows.append(
                {
                    "budget": b,
                    "strategy": strategy,
                    "steps": int(h.sum()),
                    "h_median": float(np.median(h)),
                    **_metrics(f_hat, real),
                }
            )
    return rows


def _calibrated_estimate(
    trajs: list[Trajectory], h: int, rng: np.random.Generator, eps_frac: float = 0.1
) -> np.ndarray:
    """Undiscounted-scale estimate whose tail is affinely calibrated on a subset.

    The calibration set stands in for the epsilon-sampled individuals that the
    production gate already rolls out in full, so the fit uses no oracle
    information beyond what SC-ERL already pays for.
    """
    n = len(trajs)
    partial = np.array([t.rewards[: min(h, t.length)].sum() for t in trajs])
    alive = np.array([h < t.length for t in trajs], dtype=float)
    q_h = np.array(
        [t.q[:, min(h, t.length - 1)].mean() if h < t.length else 0.0 for t in trajs]
    )
    feature = alive * (T_MAX - h) * (1.0 - GAMMA) * q_h

    cal = rng.choice(n, size=max(2, int(eps_frac * n)), replace=False)
    remaining = np.array([trajs[j].real for j in cal]) - partial[cal]
    design = np.vstack([feature[cal], alive[cal]]).T
    coef, *_ = np.linalg.lstsq(design, remaining, rcond=None)
    return partial + coef[0] * feature + coef[1] * alive


def sweep(trajs: list[Trajectory], tail: str) -> list[dict]:
    """Fixed H for every member: isolates the estimator from the allocation rule."""
    real = np.array([t.real for t in trajs])
    full_cost = int(sum(t.length for t in trajs))
    rng = np.random.default_rng(0)
    rows = []
    for h in SWEEP_H:
        cost = int(sum(min(h, t.length) for t in trajs))
        if tail == "cal":
            f_hat = _calibrated_estimate(trajs, h, rng)
        else:
            f_hat = np.array([t.estimate(h, tail)[0] for t in trajs])
        rows.append(
            {
                "h": h,
                "cost_frac": cost / full_cost,
                **_metrics(f_hat, real),
            }
        )
    return rows


def _cal_fit(
    trajs: list[Trajectory], h: np.ndarray, cal_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated undiscounted estimate and its ensemble spread, for a per-member H."""
    partial = np.array(
        [t.rewards[: min(int(h[j]), t.length)].sum() for j, t in enumerate(trajs)]
    )
    alive = np.array(
        [1.0 if int(h[j]) < t.length else 0.0 for j, t in enumerate(trajs)]
    )
    idx = [min(int(h[j]), t.length - 1) for j, t in enumerate(trajs)]
    q_mean = np.array([t.q[:, idx[j]].mean() for j, t in enumerate(trajs)])
    q_std = np.array([t.q[:, idx[j]].std(ddof=1) for j, t in enumerate(trajs)])
    scale = alive * (T_MAX - np.minimum(h, [t.length for t in trajs])) * (1.0 - GAMMA)

    # Calibration members are rolled out in full, so their tail feature at their
    # own horizon is zero by construction. Evaluate them at the horizons the
    # free members actually use instead, where the feature carries signal.
    free = np.setdiff1d(np.arange(len(trajs)), cal_idx)
    design, target = [], []
    for hh in np.unique(h[free].astype(int)):
        for c in cal_idx:
            t = trajs[c]
            if hh >= t.length:
                continue
            feat = (T_MAX - hh) * (1.0 - GAMMA) * t.q[:, hh].mean()
            design.append([feat, 1.0])
            target.append(t.real - t.rewards[:hh].sum())

    if len(design) < 2:
        coef = np.array([1.0, 0.0])
    else:
        coef, *_ = np.linalg.lstsq(np.array(design), np.array(target), rcond=None)

    f_hat = partial + coef[0] * scale * q_mean + coef[1] * alive
    sigma = np.abs(coef[0]) * scale * q_std
    return f_hat, sigma


def _allocate_cal(
    trajs: list[Trajectory],
    budget: int,
    strategy: str,
    cal_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Spend `budget` on non-calibration members; calibration members go full."""
    n = len(trajs)
    lengths = np.array([t.length for t in trajs])
    h = np.zeros(n, dtype=int)
    h[cal_idx] = lengths[cal_idx]
    free = np.setdiff1d(np.arange(n), cal_idx)
    if strategy == "uniform":
        h[free] = np.minimum(budget // len(free), lengths[free])
        return h

    h[free] = np.minimum(min(H_MIN, budget // len(free)), lengths[free])
    spent = int(h[free].sum())
    if strategy == "random_gate":
        for j in rng.permutation(free):
            extra = lengths[j] - h[j]
            if spent + extra > budget:
                continue
            h[j], spent = lengths[j], spent + extra
        return h

    while spent < budget:
        f_hat, sigma = _cal_fit(trajs, h, cal_idx)
        open_mask = np.zeros(n, dtype=bool)
        open_mask[free] = (h[free] < lengths[free]) & (sigma[free] > 0)
        if not open_mask.any():
            break
        if strategy == "sigma_greedy":
            score = np.where(open_mask, sigma, -np.inf)
        else:
            f_cut = np.sort(f_hat)[-ELITE_K]
            p = norm.cdf(-np.abs(f_hat - f_cut) / np.maximum(sigma, 1e-8))
            score = np.where(open_mask, p, -np.inf)
        j = int(np.argmax(score))
        step = int(min(CHUNK, lengths[j] - h[j], budget - spent))
        h[j], spent = h[j] + step, spent + step
    return h


def simulate_cal(trajs: list[Trajectory], seed: int = 0) -> list[dict]:
    """Allocation comparison on the calibrated estimator, epsilon cost included."""
    rng = np.random.default_rng(seed)
    n = len(trajs)
    real = np.array([t.real for t in trajs])
    lengths = np.array([t.length for t in trajs])
    full_cost = int(lengths.sum())
    cal_idx = rng.choice(n, size=max(2, n // 10), replace=False)
    cal_cost = int(lengths[cal_idx].sum())

    rows = []
    for b in BUDGETS:
        free_budget = int(b * full_cost) - cal_cost
        if free_budget <= 0:
            continue
        for strategy in STRATEGIES:
            h = _allocate_cal(trajs, free_budget, strategy, cal_idx, rng)
            f_hat, _ = _cal_fit(trajs, h, cal_idx)
            f_hat[cal_idx] = real[cal_idx]
            rows.append(
                {
                    "budget": b,
                    "strategy": strategy,
                    "steps": int(np.minimum(h, lengths).sum()),
                    "h_median": float(np.median(h)),
                    **_metrics(f_hat, real),
                }
            )
    return rows


def _print_sweep(rows: list[dict]) -> None:
    print(f"  {'H':>5} | {'cost':>6} | {'spearman':>8} | {'top10':>5} | {'regret':>7}")
    for r in rows:
        print(
            f"  {r['h']:>5} | {r['cost_frac']:>6.2f} | {r['spearman']:>8.3f} | "
            f"{r['top10_prec']:>5.2f} | {r['regret']:>7.2f}"
        )


def _print(ck: dict, trajs: list[Trajectory], rows: list[dict], tail: str) -> None:
    lengths = [t.length for t in trajs]
    real = np.array([t.real for t in trajs])
    print(
        f"\n=== {ck['env_id']} step={ck['total_steps']} tail={tail} | n_pop={len(trajs)} "
        f"ep_len median={np.median(lengths):.0f} full_cost={sum(lengths)} "
        f"| real mean={real.mean():.1f} std={real.std():.1f} ==="
    )
    print(
        f"{'budget':>7} | {'strategy':>16} | {'steps':>6} | {'H_med':>6} | {'spearman':>8} | {'top10':>5} | {'regret':>7}"
    )
    for r in rows:
        print(
            f"{r['budget']:>7.2f} | {r['strategy']:>16} | {r.get('steps', 0):>6} | "
            f"{r.get('h_median', 0):>6.0f} | {r['spearman']:>8.3f} | {r['top10_prec']:>5.2f} | {r['regret']:>7.2f}"
        )


def _online_estimate(
    calibrator: TailCalibrator, traj: Trajectory, h: int
) -> tuple[float, float]:
    idx = min(h, traj.length - 1)
    return calibrator.estimate(
        partial_return=float(traj.rewards[:h].sum()),
        h=h,
        q_mean=float(traj.q[:, idx].mean()),
        q_std=float(traj.q[:, idx].std(ddof=1)),
        alive=h < traj.length,
    )


def _check(name: str, got: float, expected: float) -> None:
    if not np.isclose(got, expected):
        raise SystemExit(
            f"{name}: online estimator gives {got!r}, reference gives {expected!r}. "
            "src/common/h_bootstrap.py and this script have drifted apart"
        )


def verify(ck: dict, trajs: list[Trajectory]) -> None:
    """Cross-check the production estimator against the offline reference."""
    rng = np.random.default_rng(0)
    n = len(trajs)
    cal_idx = rng.choice(n, size=max(2, n // 10), replace=False)

    # mirrors the training loop, where the chunk shrinks with the budget and so
    # keeps harvesting enough pairs on short-episode envs
    median_len = int(np.median([t.length for t in trajs]))
    chunk = max(1, min(CHUNK, median_len // 10))

    calibrator = TailCalibrator(gamma=GAMMA, horizon=T_MAX)
    for j in cal_idx:
        calibrator.add_trajectory(trajs[j].rewards, trajs[j].q.mean(axis=0), chunk)
    calibrator.fit()

    for traj in trajs:
        f_full, sigma_full = _online_estimate(calibrator, traj, traj.length)
        _check("full-H estimate", f_full, traj.real)
        _check("full-H sigma", sigma_full, 0.0)

        h = min(chunk, traj.length - 1)
        f_hat, sigma = _online_estimate(calibrator, traj, h)
        scale = (T_MAX - h) * (1.0 - GAMMA)
        _check(
            "affine tail",
            f_hat,
            float(traj.rewards[:h].sum())
            + calibrator.a * scale * float(traj.q[:, h].mean())
            + calibrator.b,
        )
        _check(
            "tail sigma",
            sigma,
            abs(calibrator.a) * scale * float(traj.q[:, h].std(ddof=1)),
        )

    real = np.array([t.real for t in trajs])
    print(
        f"\n=== verify {ck['env_id']} step={ck['total_steps']} | "
        f"a={calibrator.a:.4f} b={calibrator.b:.2f} pairs={calibrator.n_pairs} ==="
    )
    print(f"  {'H':>5} | {'spearman online':>15} | {'spearman offline':>16}")
    for h in SWEEP_H:
        online = np.array([_online_estimate(calibrator, t, h)[0] for t in trajs])
        offline = _calibrated_estimate(trajs, h, np.random.default_rng(0))
        print(
            f"  {h:>5} | {spearmanr(online, real).correlation:>15.3f} | "
            f"{spearmanr(offline, real).correlation:>16.3f}"
        )


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    torch.manual_seed(0)

    if paths[0] == "--verify":
        for path in paths[1:]:
            ck, trajs = collect(path)
            verify(ck, trajs)
        return
    for p in paths:
        ck, trajs = collect(p)
        for tail in ("disc", "lin"):
            _print(ck, trajs, simulate(trajs, tail), tail)
        real = np.array([t.real for t in trajs])
        print(
            f"\n### {ck['env_id']} step={ck['total_steps']} fixed-H sweep "
            f"| ep_len median={np.median([t.length for t in trajs]):.0f} "
            f"| real mean={real.mean():.1f} std={real.std():.1f}"
        )
        for tail in ("disc", "lin", "cal"):
            print(f"  -- tail={tail} --")
            _print_sweep(sweep(trajs, tail))
        print(f"\n### {ck['env_id']} step={ck['total_steps']} allocation on tail=cal")
        _print(ck, trajs, simulate_cal(trajs), "cal")


if __name__ == "__main__":
    main()
