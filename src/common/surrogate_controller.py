import copy
import warnings
from collections import deque
from enum import Enum

import numpy as np
import torch
from scipy.stats.mstats import spearmanr

from common.h_bootstrap import EpisodeRunner, TailCalibrator, stop_probability
from common.reply_buffer import Buffer
from common.utils import (
    auc_score,
    behavioural_distance,
    rollout_policy,
    surrogate_fitness,
)
from modules.deep_modules import Actor, AdaptiveBeta, Critic, EvidentialCritic
from modules.ensemble_module import EnsembleModule
from modules.evolution_module import EvolutionModule
from modules.mc_dropout_module import MCDropout

EPS_EVIDENTIAL = 1e-6
POSINF_CLAMP = 1e3
LCB_FLOOR = -5000.0
CV_POWER = 0.5
CV_OFFSET = 1.0
MIN_BUFFER_FOR_ESTIMATE = 1
EPS_POOL_SIZE = 500
EPS_POOL_MIN = 30


class SurrogateMode(Enum):
    RANDOM = 1
    DROPOUT = 2
    ENSEMBLE = 3
    EVIDENTIAL = 4

    @staticmethod
    def to_mode(str_value: str):
        str_value = str_value.lower()
        if str_value == "random":
            return SurrogateMode.RANDOM
        elif str_value == "dropout":
            return SurrogateMode.DROPOUT
        elif str_value == "ensemble":
            return SurrogateMode.ENSEMBLE
        elif str_value == "evidential":
            return SurrogateMode.EVIDENTIAL
        else:
            raise ValueError(f"Invalid surrogate mode: {str_value}")


class SurrogateController:
    def __init__(
        self,
        evolution_module: EvolutionModule,
        critic: Critic | EnsembleModule | EvidentialCritic,
        replay_buffer: Buffer,
        device: torch.device,
        surrogate_mode: SurrogateMode = SurrogateMode.RANDOM,
        omega: float = 0.5,
        rng: np.random.Generator | None = None,
        k: int = 25000,
        beta: float = 1.0,
        dropout_p: float = 0.2,
        mc_samples: int = 20,
        epsilon: float = 0.2,
        crossover_prob: float = 0.0,
        crossover_mode: str = "none",
        mad_k: float = 2.483,
        beta_lr: float = 1e-3,
        debug: bool = False,
        gate_mode: str = "topk",
        rho: float = 0.10,
        rho_min: float = 0.05,
        rho_max: float = 0.5,
        rho_eta: float = 4,
        e_star: float = 0.25,
        e_hat_window: int = 10,
        fitness_norm: str = "clip",
        gamma: float = 0.99,
        horizon: int | None = None,
        h_chunk: int = 25,
        h_alloc: str = "adaptive",
        p_stop: float = 0.05,
    ):
        if gate_mode not in ("topk", "relative", "h_bootstrap"):
            raise ValueError(f"unknown gate_mode: {gate_mode!r}")
        if h_alloc not in ("adaptive", "uniform"):
            raise ValueError(f"unknown h_alloc: {h_alloc!r}")
        if fitness_norm not in ("tanh", "clip"):
            raise ValueError(f"unknown fitness_norm: {fitness_norm!r}")
        if k < 1000:
            warnings.warn(
                f"surrogate batch k={k} is very small; use k>=1000 for "
                "stable uncertainty/LCB estimates"
            )

        self.debug = debug
        self.evolution_module = evolution_module
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.device = device
        self.omega = omega
        self.rng = rng if rng is not None else np.random.default_rng()
        self.k = k
        self.mad_k = mad_k

        self.gate_mode = gate_mode
        self.rho = rho
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_eta = rho_eta
        self.e_star = e_star
        self.e_hat_window = e_hat_window
        self._e_hat_history: list[float] = []
        self._e_hat_generation_count = 0

        self.fitness_norm = fitness_norm

        self.h_chunk = h_chunk
        self.h_alloc = h_alloc
        self.p_stop = p_stop
        self.calibrator = TailCalibrator(gamma=gamma, horizon=horizon)
        self._episode_len_ema: float | None = None
        self.last_f_cut: float | None = None
        self.last_h: list[int] = []
        self.last_sigma_tail: list[float] = []
        self.last_budget: int = 0
        self.last_n_full: int = 0

        self.adaptive_beta = AdaptiveBeta(init_value=beta).to(device)
        self._beta_optimizer = torch.optim.Adam(
            self.adaptive_beta.parameters(), lr=beta_lr
        )

        self.dropout_p = dropout_p
        self.mc_samples = mc_samples
        self.epsilon = epsilon
        self.crossover_prob = crossover_prob
        self.crossover_mode = crossover_mode
        self.last_fitness = []
        self.last_real_fitness: list[float | None] = []
        self.last_uncertainty = []
        self.last_uncertainty_mean = 0.0
        self.last_uncertainty_max = 0.0
        self.last_uncertainty_threshold = 0.0
        self.last_mu_mean = 0.0
        self.mode = "real"
        self.surrogate_mode = surrogate_mode

        self._running_fitness_min: float | None = None
        self._running_fitness_max: float | None = None
        self._fitness_ema_alpha: float = 0.05

        self._running_real_min: float | None = None
        self._running_real_max: float | None = None

        self.best_real_actor_state = None
        self.best_real_fitness = -float("inf")

        self.last_elite_indices: list[int] = []
        self.last_unselect_indices: list[int] = []
        self.last_surrogate_ratio: float = 0.0
        self.last_n_real: int = 0
        self.last_raw_sigma_mean: float = 0.0
        self.last_raw_sigma_max: float = 0.0
        self.last_raw_sigma_std: float = 0.0
        self._generation: int = 0

        self.last_gate_deterministic: list[bool] = []
        self.last_gate_epsilon: list[bool] = []
        self.last_gate_quality: dict[str, float] | None = None
        self._eps_u: deque[float] = deque(maxlen=EPS_POOL_SIZE)
        self._eps_e: deque[float] = deque(maxlen=EPS_POOL_SIZE)
        self._eps_d: deque[float] = deque(maxlen=EPS_POOL_SIZE)

        self.last_per_state_mu: np.ndarray | None = None
        self.last_per_state_sigma: np.ndarray | None = None
        self.last_obs_batch: np.ndarray | None = None
        self.last_behavioral_distance: list[float] | None = None

    @property
    def e_hat_mean(self) -> float:
        if not self._e_hat_history:
            return float("nan")
        return float(np.mean(self._e_hat_history))

    @property
    def raw_sigma_cv(self) -> float:
        if self.last_raw_sigma_mean <= 1e-12:
            return float("nan")
        return self.last_raw_sigma_std / self.last_raw_sigma_mean

    def generation_based_control(
        self,
        population: list[Actor],
        env,
        evaluate_episodes: int = 5,
        mutation_std: float = 0.05,
        mutation_prob: float = 0.1,
        elite_ratio: float = 0.2,
        total_steps: int = 0,
        warmup_steps: int = 0,
        mutation_fraction: float = 0.1,
        actor: Actor | None = None,
        action_limit: float = 1.0,
    ) -> tuple[list[Actor], list[float], int, bool]:
        self.last_uncertainty = []
        self._generation += 1

        if total_steps < warmup_steps:
            if self.gate_mode == "h_bootstrap":
                fitnesses, steps = self._h_warmup_evaluation(population, env)
            else:
                fitnesses, steps = self._real_evaluation(
                    population, env, evaluate_episodes
                )
            self.last_fitness = fitnesses
            self.last_real_fitness = list(fitnesses)
            self.mode = "real"

            if self.surrogate_mode is not SurrogateMode.RANDOM:
                if len(self.replay_buffer) < MIN_BUFFER_FOR_ESTIMATE:
                    self.last_uncertainty = []
                else:
                    obs = self._sample_obs()
                    mu, sigma, mu_per_state, sigma_per_state = self._estimate(
                        self.critic, population, obs
                    )
                    self._record_estimate(sigma, mu_per_state, sigma_per_state, obs)
                    self._update_uncertainty_metrics(self._cv(mu, sigma))

            population, fitnesses = self._evolve_and_anchor(
                population,
                fitnesses,
                mutation_std,
                mutation_prob,
                elite_ratio,
                mutation_fraction,
                allow_anchor=True,
            )
            self.last_surrogate_ratio = 0.0
            self.last_n_real = len(population)
            return population, fitnesses, steps, True

        surrogate_critic = copy.deepcopy(self.critic)
        surrogate_critic.eval()

        surrogate_count = 0

        if self.gate_mode == "h_bootstrap":
            fitnesses, steps, surrogate_count = self._h_bootstrap_evaluation(
                population=population,
                env=env,
                surrogate_critic=surrogate_critic,
                elite_ratio=elite_ratio,
                actor=actor,
                action_limit=action_limit,
            )
            self.last_fitness = fitnesses

        elif self.surrogate_mode == SurrogateMode.RANDOM:
            scaled_fitnesses = surrogate_fitness(
                population, surrogate_critic, self.replay_buffer, self.device, self.k
            )

            fitnesses = []
            real_fitness: list[float | None] = []
            steps = 0

            for i, policy in enumerate(population):
                if self.rng.random() > self.omega:
                    fit, s = self._real_evaluation(
                        [policy], env, evaluate_episodes, update_bounds=False
                    )
                    fitnesses.append(fit[0])
                    real_fitness.append(fit[0])
                    steps += s
                else:
                    fitnesses.append(scaled_fitnesses[i])
                    real_fitness.append(None)
                    surrogate_count += 1

            self.last_fitness = fitnesses
            self.last_real_fitness = real_fitness
            self._update_real_bounds([f for f in real_fitness if f is not None])

        elif self.surrogate_mode in (
            SurrogateMode.DROPOUT,
            SurrogateMode.ENSEMBLE,
            SurrogateMode.EVIDENTIAL,
        ):
            obs = self._sample_obs()
            mu, sigma, mu_per_state, sigma_per_state = self._estimate(
                surrogate_critic, population, obs
            )
            self._record_estimate(sigma, mu_per_state, sigma_per_state, obs)
            self.last_behavioral_distance = (
                [
                    behavioural_distance(policy, actor, obs, action_limit)
                    for policy in population
                ]
                if actor is not None
                else None
            )
            cv_values = self._cv(mu, sigma)
            threshold = self._update_uncertainty_metrics(cv_values)

            fitnesses, steps, surrogate_count = self._gated_evaluation(
                population, env, evaluate_episodes, mu, sigma, cv_values, threshold
            )
            self.last_fitness = fitnesses

        else:
            raise ValueError(f"unhandled surrogate mode: {self.surrogate_mode}")

        self.last_surrogate_ratio = (
            surrogate_count / len(population) if population else 0.0
        )
        self.last_n_real = len(population) - surrogate_count
        self.mode = "surrogate" if surrogate_count > 0 else "real"
        used_real_eval = surrogate_count < len(population)

        population, fitnesses = self._evolve_and_anchor(
            population,
            fitnesses,
            mutation_std,
            mutation_prob,
            elite_ratio,
            mutation_fraction,
            allow_anchor=used_real_eval,
        )

        return population, fitnesses, steps, used_real_eval

    def _estimate(
        self,
        critic: Critic | EnsembleModule | EvidentialCritic,
        population: list[Actor],
        obs: torch.Tensor,
    ) -> tuple[list[float], list[float], list[np.ndarray], list[np.ndarray]]:
        if self.surrogate_mode == SurrogateMode.DROPOUT:
            return MCDropout.fitness_evaluation_mc_dropout(
                critic=self._as_dropout_critic(critic),
                population=population,
                obs=obs,
                device=self.device,
                T=self.mc_samples,
                dropout_p=self.dropout_p,
            )

        mu_out, sigma_out, mu_per_state, sigma_per_state = [], [], [], []

        for policy in population:
            mu, sigma = self._q_stats(critic, policy, obs)
            mu_out.append(float(mu.mean()))
            sigma_out.append(float(sigma.mean()))
            mu_per_state.append(mu)
            sigma_per_state.append(sigma)

        return mu_out, sigma_out, mu_per_state, sigma_per_state

    def _as_dropout_critic(
        self, critic: Critic | EnsembleModule | EvidentialCritic
    ) -> Critic:
        if not isinstance(critic, Critic):
            raise TypeError(
                f"surrogate.mode=dropout requires a plain Critic, got "
                f"{type(critic).__name__}"
            )
        return critic

    def _q_stats(
        self,
        critic: Critic | EnsembleModule | EvidentialCritic,
        policy: Actor,
        obs: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.surrogate_mode == SurrogateMode.DROPOUT:
            _, _, mu_per_state, sigma_per_state = (
                MCDropout.fitness_evaluation_mc_dropout(
                    critic=self._as_dropout_critic(critic),
                    population=[policy],
                    obs=obs,
                    device=self.device,
                    T=self.mc_samples,
                    dropout_p=self.dropout_p,
                )
            )
            return mu_per_state[0], sigma_per_state[0]

        critic.eval()
        policy.eval()
        with torch.no_grad():
            actions = policy(obs)
            if self.surrogate_mode == SurrogateMode.EVIDENTIAL:
                mu, v, alpha, beta = critic(obs, actions)
                var = beta / (v * (alpha - 1.0) + EPS_EVIDENTIAL)
                var = torch.nan_to_num(var, nan=0.0, posinf=POSINF_CLAMP, neginf=0.0)
                sigma = torch.sqrt(var.clamp(min=0.0))
            elif self.surrogate_mode == SurrogateMode.RANDOM:
                mu = critic(obs, actions)
                mu = mu[0] if isinstance(mu, tuple) else mu
                sigma = torch.zeros_like(mu)
            else:
                mu, sigma = critic(obs, actions)

        return mu.squeeze(-1).cpu().numpy(), sigma.squeeze(-1).cpu().numpy()

    def _record_estimate(
        self,
        sigma: list[float],
        mu_per_state: list[np.ndarray],
        sigma_per_state: list[np.ndarray],
        obs: torch.Tensor,
    ) -> None:
        self.last_uncertainty = sigma
        self.last_obs_batch = obs.detach().cpu().numpy()
        self.last_per_state_mu = np.stack(mu_per_state, axis=0)
        self.last_per_state_sigma = np.stack(sigma_per_state, axis=0)
        self.last_mu_mean = float(self.last_per_state_mu.mean())

        if self.debug:
            assert np.allclose(
                self.last_per_state_sigma.mean(axis=1), np.array(sigma)
            ), "per-state sigma mean diverges from scalar last_uncertainty"

    def _sample_obs(self) -> torch.Tensor:
        k = min(self.k, len(self.replay_buffer))
        return self.replay_buffer.sample(batch_size=k)["state"].to(self.device)

    def _lcb(
        self,
        mu: list[float],
        sigma: list[float],
        lcb_floor: float = LCB_FLOOR,
    ) -> list[float]:
        b = self.adaptive_beta.beta
        return [
            float(np.clip(m - b * s, a_min=lcb_floor, a_max=None))
            for m, s in zip(mu, sigma)
        ]

    def _cv(
        self,
        mu: list[float],
        sigma: list[float],
        power: float = CV_POWER,
        offset: float = CV_OFFSET,
    ) -> list[float]:
        return [s / (abs(m) ** power + offset) for s, m in zip(sigma, mu)]

    def _gate_mask(self, u, cv, threshold):
        n = len(u)
        if self.gate_mode == "relative":
            mask = np.array([c > threshold for c in cv])
            return mask, np.zeros(n, dtype=bool)

        m = min(max(1, int(np.ceil(self.rho * n))), n)

        n_eps = int(np.clip(round(self.epsilon * m), 0, max(0, m - 1)))
        n_rank = m - n_eps

        order = np.argsort(-np.asarray(u))
        chosen = list(order[:n_rank])
        rest = list(order[n_rank:])
        if n_eps > 0 and rest:
            picked = self.rng.choice(rest, size=min(n_eps, len(rest)), replace=False)
            chosen += list(picked)

        mask = np.zeros(n, dtype=bool)
        is_eps = np.zeros(n, dtype=bool)
        mask[chosen] = True
        is_eps[chosen[n_rank:]] = True
        return mask, is_eps

    def _update_rho(self, e_hat: float) -> None:
        self.rho = float(
            np.clip(
                self.rho + self.rho_eta * (e_hat - self.e_star),
                self.rho_min,
                self.rho_max,
            )
        )

    def _surrogate_error(self, min_n=20):
        if len(self._eps_e) < min_n:
            return None
        if self._running_real_min is None or self._running_real_max is None:
            return None
        scale = self._running_real_max - self._running_real_min
        if scale < 1e-8:
            return None
        return float(
            np.mean(list(self._eps_e)[-40:]) / scale
        )  # 40 pairs equals 20 generation - EMA correctness

    def _update_gate_quality(
        self, scaled: list[float], fitnesses: list[float | None]
    ) -> None:
        for i, is_eps in enumerate(self.last_gate_epsilon):
            fitness_i = fitnesses[i]
            if is_eps and fitness_i is not None:
                self._eps_u.append(self.last_uncertainty[i])
                self._eps_e.append(abs(scaled[i] - fitness_i))
                if self.last_behavioral_distance is not None:
                    self._eps_d.append(self.last_behavioral_distance[i])

        self._recompute_gate_quality()

    def _recompute_gate_quality(self) -> None:
        if len(self._eps_u) < EPS_POOL_MIN:
            self.last_gate_quality = None
            return

        u = np.array(self._eps_u)
        e = np.array(self._eps_e)
        y = (e > np.median(e)).astype(int)
        self.last_gate_quality = {
            "auc_u": auc_score(u, y),
            "spearman": float(spearmanr(u, e).correlation),
            "n_pool": float(len(u)),
        }

        has_distance_pool = len(self._eps_d) == len(self._eps_u)
        if has_distance_pool:
            d = np.array(self._eps_d)
            self.last_gate_quality["auc_d"] = auc_score(d, y)

    def _accumulate_rho_error(self) -> None:
        if self.gate_mode != "topk":
            return
        e_hat = self._surrogate_error()
        if e_hat is not None:
            self._e_hat_history.append(e_hat)

        self._e_hat_generation_count += 1
        if self._e_hat_generation_count >= self.e_hat_window:
            if self._e_hat_history:
                self._update_rho(float(np.mean(self._e_hat_history)))
                self._e_hat_history = []
            self._e_hat_generation_count = 0

    def _gated_evaluation(
        self,
        population: list[Actor],
        env,
        episodes: int,
        mu: list[float],
        sigma: list[float],
        cv: list[float],
        threshold: float,
    ) -> tuple[list[float], int, int]:
        scaled = self._normalize_surrogate_fitness(self._lcb(mu, sigma))

        mask, is_eps = self._gate_mask(sigma, cv, threshold)

        fitnesses, steps, surrogate_count = [], 0, 0
        beta_mu, beta_sigma, beta_real = [], [], []
        self.last_gate_deterministic = [
            bool(m) and not bool(e) for m, e in zip(mask, is_eps)
        ]
        self.last_gate_epsilon = [bool(e) for e in is_eps]
        self.last_real_fitness = [None] * len(population)

        for i, policy in enumerate(population):
            if mask[i]:
                fit, s = self._real_evaluation(
                    [policy], env, episodes, update_bounds=False
                )
                fitnesses.append(fit[0])
                self.last_real_fitness[i] = fit[0]
                beta_real.append(fit[0])
                beta_mu.append(mu[i])
                beta_sigma.append(sigma[i])
                steps += s
            else:
                fitnesses.append(scaled[i])
                surrogate_count += 1

        self._update_real_bounds([f for f in self.last_real_fitness if f is not None])
        self._update_beta(beta_mu, beta_sigma, beta_real)
        self._accumulate_rho_error()
        self._update_gate_quality(scaled, self.last_real_fitness)
        return fitnesses, steps, surrogate_count

    def _h_budget(self, n_pop: int) -> int:
        return int(self.rho * n_pop * self.episode_length)

    @property
    def episode_length(self) -> float:
        return self._episode_len_ema or self.calibrator.horizon

    def _h_eps_count(self, n_pop: int, budget: int) -> int:
        if not self.calibrator.calibrated:
            # no pairs to fit the tail on yet, so half the budget buys the full
            # rollouts the affine fit needs
            return int(np.clip(0.5 * budget / self.episode_length, 1, n_pop))
        return int(np.clip(round(self.epsilon * self.rho * n_pop), 1, n_pop))

    def _h_chunk_eff(self, n_pop: int, budget: int, eps_count: int) -> int:
        free = budget - eps_count * self.episode_length
        return int(np.clip(free // n_pop, 1, self.h_chunk))

    def _h_order(
        self, mu: list[float], sigma: list[float], elite_count: int
    ) -> list[int]:
        # cheap decisions first: individuals far from the elite threshold stop
        # after one chunk, which leaves the budget — and a well estimated F_cut
        # — for the ones sitting on the threshold
        prior = np.array(self._lcb(mu, sigma))
        cut = float(np.sort(prior)[-elite_count])
        risk = [stop_probability(p, s, cut) for p, s in zip(prior, sigma)]
        return [int(i) for i in np.argsort(risk)]

    def _h_prior_order(
        self,
        population: list[Actor],
        critic: Critic | EnsembleModule | EvidentialCritic,
        *,
        elite_count: int,
        actor: Actor | None,
        action_limit: float,
    ) -> list[int]:
        if self.surrogate_mode == SurrogateMode.RANDOM:
            return [int(i) for i in self.rng.permutation(len(population))]

        obs = self._sample_obs()
        mu, sigma, mu_per_state, sigma_per_state = self._estimate(
            critic, population, obs
        )
        self._record_estimate(sigma, mu_per_state, sigma_per_state, obs)
        self.last_behavioral_distance = (
            [
                behavioural_distance(policy, actor, obs, action_limit)
                for policy in population
            ]
            if actor is not None
            else None
        )
        self._update_uncertainty_metrics(self._cv(mu, sigma))
        return self._h_order(mu, sigma, elite_count)

    def _h_estimate(
        self,
        critic: Critic | EnsembleModule | EvidentialCritic,
        policy: Actor,
        runner: EpisodeRunner,
    ) -> tuple[float, float]:
        obs = torch.as_tensor(
            runner.last_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mu, sigma = self._q_stats(critic, policy, obs)
        return self.calibrator.estimate(
            partial_return=runner.cum_reward,
            h=runner.steps,
            q_mean=float(mu[0]),
            q_std=float(sigma[0]),
            alive=not runner.done,
        )

    def _roll_individual(
        self,
        policy: Actor,
        env,
        *,
        critic: Critic | EnsembleModule | EvidentialCritic,
        chunk: int,
        h_cap: int | None,
        f_cut: float | None,
    ) -> tuple[float, float, EpisodeRunner]:
        # h_cap None marks an epsilon individual: rolled to the end of the
        # episode whatever the stop rule says, because only a complete return
        # can calibrate the tail and measure the surrogate error
        is_epsilon = h_cap is None
        runner = EpisodeRunner(
            policy,
            env,
            self.device,
            self.replay_buffer,
            store_states=is_epsilon,
        )
        if is_epsilon:
            runner.advance()
            return runner.cum_reward, 0.0, runner

        f_hat, sigma = 0.0, 0.0
        while not runner.done and runner.steps < h_cap:
            runner.advance(min(chunk, h_cap - runner.steps))
            f_hat, sigma = self._h_estimate(critic, policy, runner)
            if (
                f_cut is not None
                and stop_probability(f_hat, sigma, f_cut) < self.p_stop
            ):
                break
        return f_hat, sigma, runner

    def _harvest_full(
        self,
        policy: Actor,
        runner: EpisodeRunner,
        *,
        critic: Critic | EnsembleModule | EvidentialCritic,
        chunk: int,
    ) -> tuple[list[float], list[float], list[float]]:
        states = torch.as_tensor(
            np.array(runner.states), dtype=torch.float32, device=self.device
        )
        mu, sigma = self._q_stats(critic, policy, states)
        rewards = np.asarray(runner.rewards)
        self.calibrator.add_trajectory(rewards, mu, chunk)

        if not self.calibrator.calibrated:
            return [], [], []

        total = float(rewards.sum())
        f_hats, sigmas, reals = [], [], []
        for h in range(chunk, len(rewards), chunk):
            f_hat, sig = self.calibrator.estimate(
                partial_return=float(rewards[:h].sum()),
                h=h,
                q_mean=float(mu[h]),
                q_std=float(sigma[h]),
                alive=True,
            )
            self._eps_u.append(sig)
            self._eps_e.append(abs(f_hat - total))
            f_hats.append(f_hat)
            sigmas.append(sig)
            reals.append(total)
        return f_hats, sigmas, reals

    def _observe_full_episode(self, policy: Actor, runner: EpisodeRunner) -> None:
        alpha = self._fitness_ema_alpha
        if self._episode_len_ema is None:
            self._episode_len_ema = float(runner.steps)
        else:
            self._episode_len_ema = (
                alpha * runner.steps + (1 - alpha) * self._episode_len_ema
            )
        self.calibrator.observe_episode_length(runner.steps)
        self._anchor(policy, runner.cum_reward)

    def _h_warmup_evaluation(
        self, population: list[Actor], env
    ) -> tuple[list[float], int]:
        critic = copy.deepcopy(self.critic)
        critic.eval()

        fitnesses, steps = [], 0
        self.last_h, self.last_sigma_tail = [], [0.0] * len(population)
        for policy in population:
            runner = EpisodeRunner(
                policy, env, self.device, self.replay_buffer, store_states=True
            )
            runner.advance()
            fitnesses.append(runner.cum_reward)
            steps += runner.steps
            self.last_h.append(runner.steps)
            self._observe_full_episode(policy, runner)
            self._harvest_full(policy, runner, critic=critic, chunk=self.h_chunk)

        self.calibrator.fit()
        self._update_real_bounds(fitnesses)
        self.last_n_full = len(population)
        self.last_budget = steps
        return fitnesses, steps

    def _h_bootstrap_evaluation(
        self,
        population: list[Actor],
        env,
        *,
        surrogate_critic: Critic | EnsembleModule | EvidentialCritic,
        elite_ratio: float,
        actor: Actor | None,
        action_limit: float,
    ) -> tuple[list[float], int, int]:
        n_pop = len(population)
        elite_count = max(2 if n_pop >= 2 else 1, int(n_pop * elite_ratio))
        budget = self._h_budget(n_pop)
        eps_count = self._h_eps_count(n_pop, budget)
        chunk = self._h_chunk_eff(n_pop, budget, eps_count)
        eps_idx = set(self.rng.choice(n_pop, size=eps_count, replace=False).tolist())
        alloc = (
            "uniform" if self.surrogate_mode == SurrogateMode.RANDOM else self.h_alloc
        )
        order = self._h_prior_order(
            population,
            surrogate_critic,
            elite_count=elite_count,
            actor=actor,
            action_limit=action_limit,
        )

        fitnesses = [0.0] * n_pop
        self.last_h = [0] * n_pop
        self.last_sigma_tail = [0.0] * n_pop
        self.last_real_fitness = [None] * n_pop
        beta_mu, beta_sigma, beta_real = [], [], []
        full_returns: list[float] = []
        estimates: list[float] = []

        # the epsilon individuals roll to the end whatever happens, so their cost
        # is reserved up front and only the rest compete for what is left
        budget_left = max(0, budget - int(eps_count * self.episode_length))
        steps = 0
        f_cut = self.last_f_cut
        for position, index in enumerate(order):
            h_cap = self._h_cap(
                index in eps_idx,
                # without a threshold there is nothing to race towards, so the
                # first generation spreads the budget evenly instead of handing
                # it to whoever comes first
                alloc=alloc if f_cut is not None else "uniform",
                chunk=chunk,
                budget=budget,
                budget_left=budget_left,
                n_pop=n_pop,
                remaining=len(order) - position - 1,
            )
            f_hat, sigma, runner = self._roll_individual(
                population[index],
                env,
                critic=surrogate_critic,
                chunk=chunk,
                h_cap=h_cap,
                f_cut=f_cut if alloc == "adaptive" else None,
            )
            steps += runner.steps
            if h_cap is not None:
                budget_left -= runner.steps
            fitnesses[index] = f_hat - self.adaptive_beta.beta * sigma
            self.last_h[index] = runner.steps
            self.last_sigma_tail[index] = sigma

            if runner.done:
                self.last_real_fitness[index] = runner.cum_reward
                full_returns.append(runner.cum_reward)
                self._observe_full_episode(population[index], runner)
            if runner.states and runner.done:
                mus, sigmas, reals = self._harvest_full(
                    population[index], runner, critic=surrogate_critic, chunk=chunk
                )
                beta_mu += mus
                beta_sigma += sigmas
                beta_real += reals

            estimates.append(f_hat)
            # the order front-loads individuals far from the threshold, so the
            # online cut only replaces the previous generation's once enough of
            # the population has been evaluated for the quantile to mean anything
            if len(estimates) >= max(elite_count, n_pop // 2):
                f_cut = float(np.quantile(estimates, 1.0 - elite_ratio))

        self.last_f_cut = f_cut
        self.last_budget = budget
        self.last_n_full = len(full_returns)
        self._update_real_bounds(full_returns)
        self.calibrator.fit()
        self._update_beta(beta_mu, beta_sigma, beta_real)
        self._accumulate_rho_error()
        self._recompute_gate_quality()
        return fitnesses, steps, n_pop - len(full_returns)

    def _h_cap(
        self,
        is_epsilon: bool,
        *,
        alloc: str,
        chunk: int,
        budget: int,
        budget_left: int,
        n_pop: int,
        remaining: int,
    ) -> int | None:
        if is_epsilon:
            return None
        if alloc == "uniform":
            return max(chunk, budget // n_pop)
        # equal share of what is left, so steps the early (low-risk) individuals
        # give back by stopping early accumulate for the ones near the threshold
        # at the end of the order
        return max(chunk, budget_left // (remaining + 1))

    def _evolve_and_anchor(
        self,
        population: list[Actor],
        fitnesses: list[float],
        mutation_std: float,
        mutation_prob: float,
        elite_ratio: float,
        mutation_fraction: float,
        allow_anchor: bool,
    ) -> tuple[list[Actor], list[float]]:
        population, elites, unselected = self.evolution_module.evolve(
            population=population,
            fitnesses=fitnesses,
            mutation_std=mutation_std,
            mutation_prob=mutation_prob,
            elite_ratio=elite_ratio,
            crossover_prob=self.crossover_prob,
            crossover_mode=self.crossover_mode,
            replay_buffer=self.replay_buffer,
            mutation_fraction=mutation_fraction,
        )
        self.last_elite_indices = elites
        self.last_unselect_indices = unselected

        # Post-evolution elite injection (Global Anchor): restore best-ever real
        # actor into the worst slot so it can never be lost to evolution/surrogate noise.
        if allow_anchor and self.best_real_actor_state is not None and fitnesses:
            worst = int(np.argmin(fitnesses))
            if worst not in elites:
                target = worst
            elif unselected:
                target = unselected[-1]
            else:
                target = elites[-1]
            population[target].load_state_dict(self.best_real_actor_state)
            fitnesses[target] = self.best_real_fitness
            # last_fitness can be shorter than fitnesses/population when the
            # warmup path re-enters with a resized population; guard the write.
            if target < len(self.last_fitness):
                self.last_fitness[target] = self.best_real_fitness

        return population, fitnesses

    def _update_real_bounds(self, real_values: list[float]):
        if not real_values:
            return
        batch_min = float(np.min(real_values))
        batch_max = float(np.max(real_values))
        alpha = self._fitness_ema_alpha

        if self._running_real_min is None or self._running_real_max is None:
            self._running_real_min = batch_min
            self._running_real_max = batch_max
        else:
            self._running_real_min = (
                alpha * batch_min + (1 - alpha) * self._running_real_min
            )
            self._running_real_max = (
                alpha * batch_max + (1 - alpha) * self._running_real_max
            )

    def _real_evaluation(
        self,
        population: list[Actor],
        env,
        evaluate_episodes: int,
        store_in_buffer: bool = True,
        update_bounds: bool = True,
    ) -> tuple[list[float], int]:
        fitnesses = []
        total_steps = 0

        for individual in population:
            fitness, steps = rollout_policy(
                policy=individual,
                env=env,
                device=self.device,
                replay_buffer=self.replay_buffer,
                episodes=evaluate_episodes,
                noise_std=0.0,
                store_in_buffer=store_in_buffer,
            )

            self._anchor(individual, fitness)

            fitnesses.append(fitness)
            total_steps += steps

        if update_bounds:
            self._update_real_bounds(fitnesses)

        return fitnesses, total_steps

    def _anchor(self, individual: Actor, fitness: float) -> None:
        # Store the best real actor regardless of the surrogate predictions
        if fitness > self.best_real_fitness:
            self.best_real_fitness = fitness
            self.best_real_actor_state = copy.deepcopy(individual.state_dict())

    def _update_uncertainty_metrics(self, cv_values: list[float]) -> float:
        if not cv_values:
            self.last_uncertainty_mean = 0.0
            self.last_uncertainty_max = 0.0
            self.last_uncertainty_threshold = 0.0
            self.last_raw_sigma_mean = 0.0
            self.last_raw_sigma_max = 0.0
            self.last_raw_sigma_std = 0.0
            return 0.0

        cv_arr = np.nan_to_num(
            np.array(cv_values, dtype=np.float64),
            nan=0.0,
            posinf=POSINF_CLAMP,
            neginf=0.0,
        )

        median = float(np.median(cv_arr))
        mad = float(np.median(np.abs(cv_arr - median)))
        threshold = median + self.mad_k * mad

        self.last_uncertainty_mean = float(np.mean(cv_arr))
        self.last_uncertainty_max = float(np.max(cv_arr))
        self.last_uncertainty_threshold = threshold

        if self.last_uncertainty:
            raw = np.nan_to_num(
                np.array(self.last_uncertainty, dtype=np.float64),
                nan=0.0,
                posinf=POSINF_CLAMP,
                neginf=0.0,
            )
            self.last_raw_sigma_mean = float(np.mean(raw))
            self.last_raw_sigma_max = float(np.max(raw))
            self.last_raw_sigma_std = float(np.std(raw))
        else:
            self.last_raw_sigma_mean = 0.0
            self.last_raw_sigma_max = 0.0
            self.last_raw_sigma_std = 0.0

        return threshold

    def _update_beta(
        self,
        mu_scores: list[float],
        sigma_scores: list[float],
        real_scores: list[float],
    ) -> None:
        if len(mu_scores) < 2:
            return
        mu = torch.tensor(mu_scores, dtype=torch.float32, device=self.device)
        sigma = torch.tensor(sigma_scores, dtype=torch.float32, device=self.device)
        r = torch.tensor(real_scores, dtype=torch.float32, device=self.device)
        beta = torch.exp(self.adaptive_beta.log_beta)
        predicted = mu - beta * sigma
        loss = torch.nn.functional.mse_loss(predicted, r)
        self._beta_optimizer.zero_grad()
        loss.backward()
        self._beta_optimizer.step()

    def _normalize_surrogate_fitness(self, raw_fitnesses: list[float]) -> list[float]:
        alpha = self._fitness_ema_alpha
        batch_min = float(np.min(raw_fitnesses))
        batch_max = float(np.max(raw_fitnesses))

        if self._running_fitness_min is None or self._running_fitness_max is None:
            self._running_fitness_min = batch_min
            self._running_fitness_max = batch_max
        else:
            self._running_fitness_min = (
                alpha * batch_min + (1 - alpha) * self._running_fitness_min
            )
            self._running_fitness_max = (
                alpha * batch_max + (1 - alpha) * self._running_fitness_max
            )

        lo, hi = self._running_fitness_min, self._running_fitness_max
        span = hi - lo

        span = max(span, 1e-6)

        if self.fitness_norm == "tanh":
            normalised = [np.tanh((f - lo) / span * 2.0 - 1.0) for f in raw_fitnesses]
        else:
            normalised = [
                np.clip((f - lo) / span * 2.0 - 1.0, -1.0, 1.0) for f in raw_fitnesses
            ]

        if self._running_real_min is None or self._running_real_max is None:
            lo_real, hi_real = lo, hi
        else:
            lo_real, hi_real = self._running_real_min, self._running_real_max

        span_real = hi_real - lo_real
        rescaled = [((n + 1.0) / 2.0) * span_real + lo_real for n in normalised]

        return rescaled
