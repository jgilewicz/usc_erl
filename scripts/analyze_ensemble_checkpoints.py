"""Decisive test v3 — pre-registered criteria, evaluated on production checkpoints.

Pre-registered before any 500k/1M production checkpoint was analyzed (see the
ensemble-epistemic-signal debugging thread that produced these thresholds).
Do not adjust them after seeing results.

Hypothesis: sigma_cv and action_sensitivity are higher / more responsive in
environments where the ensemble surrogate gate improves SC-ERL over the
random-gate baseline (e.g. DMC dog-walk), and flat where it hurts (e.g.
classic MuJoCo Hopper/Swimmer).

CONFIRMED requires ALL of, on the final checkpoint:
  - action_sensitivity(dmc_env) > 3 * action_sensitivity(mujoco_env)
  - sigma_cv(dmc_env) > 0.05 AND sigma_cv(mujoco_env) < 0.02
  - direction consistent across every seed

REFUTED if ANY of:
  - action_sensitivity ratio between the two envs < 2x
  - sigma_cv comparable between envs
  - sign of the effect flips on any single seed

UNDETERMINED if:
  - d_cv < 0.02 in either env (population too homogeneous — GA/evolution
    problem, not a critic problem; the test isn't measuring what it should)
  - spread across seeds exceeds the between-env difference

This is attempt 1 of 3 allotted on this mechanism. If attempt 3 still has no
consistent predictor, the paper reports the empirical ranking gap (DMC dog
rank 3.40 vs 5.60; classic MuJoCo 5.40 vs 4.00) without a mechanistic
explanation — an honest "we don't know why" beats a strained one.

Usage:
    uv run --extra mujoco-envs python scripts/analyze_ensemble_checkpoints.py \
        checkpoints/ckpt_dm_control_dog-walk-v0_seed0_*.pt \
        checkpoints/ckpt_Hopper-v5_seed0_*.pt \
        ...
"""

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))

import numpy as np
import torch

from common.utils import behavioural_distance
from modules.deep_modules import Actor, Critic
from modules.ensemble_module import EnsembleModule

ACTION_SENSITIVITY_RATIO_CONFIRM = 3.0
ACTION_SENSITIVITY_RATIO_REFUTE = 2.0
SIGMA_CV_HIGH = 0.05
SIGMA_CV_LOW = 0.02
D_CV_HOMOGENEOUS = 0.02


def _load_actor(ckpt: dict, state_dict: dict) -> Actor:
    actor = Actor(
        ckpt["state_dim"],
        ckpt["action_dim"],
        ckpt["actor_hidden_dim"],
        ckpt["action_limit"],
        activation="tanh",
    )
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def analyze(ckpt_path: str) -> dict:
    # weights_only=False: checkpoint embeds obs_batch as a numpy array, and is
    # produced by our own training run, not third-party input.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs = torch.tensor(ck["obs_batch"], dtype=torch.float32)
    a_max = ck["action_limit"]

    critic = EnsembleModule(
        ensemble_size=ck["k_ensembles"],
        critic=Critic(ck["state_dim"], ck["action_dim"], dropout=0.0, activation="elu"),
        rng=np.random.default_rng(0),
    )
    critic.load_state_dict(ck["critic"])
    critic.eval()

    actor = _load_actor(ck, ck["actor"])

    with torch.no_grad():
        a_rl = actor(obs)
        a_rand = torch.empty_like(a_rl).uniform_(-a_max, a_max)
        mu_rl, _ = critic(obs, a_rl)
        mu_rand, _ = critic(obs, a_rand)

    # sigma at a_rand vs a_rl is deliberately not compared here: neither point
    # is an OOD reference for the critic (see probe_ensemble_uncertainty.py).
    action_sensitivity = (mu_rand - mu_rl).abs().mean() / mu_rl.abs().mean()

    pop = [_load_actor(ck, s) for s in ck["population"]]
    with torch.no_grad():
        sig = np.array([critic(obs, p(obs))[1].mean().item() for p in pop])
    d = np.array([behavioural_distance(p, actor, obs, a_max) for p in pop])

    return {
        "env_id": ck.get("env_id", ""),
        "seed": ck.get("seed", -1),
        "total_steps": ck.get("total_steps", -1),
        "action_sensitivity": float(action_sensitivity),
        "sigma_mean": float(sig.mean()),
        "sigma_cv": float(sig.std() / sig.mean()),
        "d_mean": float(d.mean()),
        "d_cv": float(d.std() / d.mean()),
        "corr_sigma_d": float(np.corrcoef(sig, d)[0, 1]),
        "n_pop": len(pop),
    }


def _print_table(rows: list[dict]) -> None:
    cols = [
        "env_id",
        "seed",
        "total_steps",
        "action_sensitivity",
        "sigma_cv",
        "d_cv",
        "corr_sigma_d",
        "n_pop",
    ]
    header = " | ".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:>18.4f}" if isinstance(v, float) else f"{v!s:>18}")
        print(" | ".join(cells))


def _verdict(dmc_rows: list[dict], mujoco_rows: list[dict]) -> str:
    if not dmc_rows or not mujoco_rows:
        return "UNDETERMINED: missing one of the two environments' checkpoints"

    reasons_undetermined = []
    reasons_refute = []

    for r in dmc_rows + mujoco_rows:
        if r["d_cv"] < D_CV_HOMOGENEOUS:
            reasons_undetermined.append(
                f"d_cv={r['d_cv']:.4f} < {D_CV_HOMOGENEOUS} for "
                f"{r['env_id']} seed={r['seed']} — population too homogeneous"
            )

    ratios, cv_dmc, cv_mujoco = [], [], []
    dmc_by_seed = {r["seed"]: r for r in dmc_rows}
    mujoco_by_seed = {r["seed"]: r for r in mujoco_rows}
    common_seeds = sorted(set(dmc_by_seed) & set(mujoco_by_seed))
    if not common_seeds:
        return "UNDETERMINED: no matching seeds between the two environments"

    for seed in common_seeds:
        d_row, m_row = dmc_by_seed[seed], mujoco_by_seed[seed]
        ratio = d_row["action_sensitivity"] / max(m_row["action_sensitivity"], 1e-12)
        ratios.append(ratio)
        cv_dmc.append(d_row["sigma_cv"])
        cv_mujoco.append(m_row["sigma_cv"])
        if ratio < ACTION_SENSITIVITY_RATIO_REFUTE:
            reasons_refute.append(
                f"seed={seed}: action_sensitivity ratio {ratio:.2f}x < "
                f"{ACTION_SENSITIVITY_RATIO_REFUTE}x"
            )
        if d_row["sigma_cv"] <= m_row["sigma_cv"]:
            reasons_refute.append(
                f"seed={seed}: sigma_cv not higher in DMC env "
                f"({d_row['sigma_cv']:.4f} <= {m_row['sigma_cv']:.4f})"
            )

    seed_spread = max(ratios) - min(ratios) if len(ratios) > 1 else 0.0
    between_env_diff = np.mean(ratios) - 1.0
    if len(ratios) > 1 and seed_spread > between_env_diff:
        reasons_undetermined.append(
            f"seed spread in ratio ({seed_spread:.2f}) exceeds "
            f"between-env effect ({between_env_diff:.2f})"
        )

    if reasons_undetermined:
        return "UNDETERMINED:\n  " + "\n  ".join(reasons_undetermined)
    if reasons_refute:
        return "REFUTED:\n  " + "\n  ".join(reasons_refute)

    all_confirmed = (
        min(ratios) > ACTION_SENSITIVITY_RATIO_CONFIRM
        and min(cv_dmc) > SIGMA_CV_HIGH
        and max(cv_mujoco) < SIGMA_CV_LOW
    )
    if all_confirmed:
        return (
            "CONFIRMED: action_sensitivity ratio "
            f"{min(ratios):.2f}-{max(ratios):.2f}x, "
            f"sigma_cv DMC {min(cv_dmc):.4f}-{max(cv_dmc):.4f} vs "
            f"MuJoCo {min(cv_mujoco):.4f}-{max(cv_mujoco):.4f}, "
            "consistent across all seeds"
        )
    return (
        "UNDETERMINED: neither REFUTED nor CONFIRMED thresholds cleanly met — "
        f"ratio {min(ratios):.2f}-{max(ratios):.2f}x, "
        f"sigma_cv DMC {min(cv_dmc):.4f}-{max(cv_dmc):.4f} vs "
        f"MuJoCo {min(cv_mujoco):.4f}-{max(cv_mujoco):.4f}"
    )


def main() -> None:
    ckpt_paths = sys.argv[1:]
    if not ckpt_paths:
        raise SystemExit(__doc__)

    rows = [analyze(p) for p in ckpt_paths]
    rows.sort(key=lambda r: (r["total_steps"], r["env_id"], r["seed"]))

    steps = sorted({r["total_steps"] for r in rows})
    for step in steps:
        print(f"\n=== checkpoint step {step} ===")
        _print_table([r for r in rows if r["total_steps"] == step])

    final_step = steps[-1]
    final_rows = [r for r in rows if r["total_steps"] == final_step]
    dmc_rows = [
        r for r in final_rows if "dm_control" in r["env_id"] or "dog" in r["env_id"]
    ]
    mujoco_rows = [r for r in final_rows if r not in dmc_rows]

    print(f"\n=== verdict (final checkpoint, step {final_step}) ===")
    print(_verdict(dmc_rows, mujoco_rows))


if __name__ == "__main__":
    main()
