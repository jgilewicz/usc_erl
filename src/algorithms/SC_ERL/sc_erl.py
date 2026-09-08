from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from common.reply_buffer import Buffer
from common.surrogate_controller import SurrogateController, SurrogateMode
from common.utils import (
    behavioural_distance,
    build_surrogate_metrics,
    evaluate_policy,
    print_sc_erl_debug_summary,
    rollout_policy,
    soft_update,
    td3_train_critics,
    td3_update_actor,
    warmup,
)
from common.wandb_logger import WandbLogger
from modules.deep_modules import Actor, Critic, EvidentialCritic
from modules.ensemble_module import EnsembleModule
from modules.evolution_module import EvolutionModule

_CPU_DEVICE = torch.device("cpu")
CHECKPOINT_FRACTIONS = (0.2, 0.6, 1.0)


def SC_ERL(
    population_size: int,
    buffer_size: int,
    rng: np.random.Generator,
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 64,
    device: torch.device = _CPU_DEVICE,
    actor_hidden_dim: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    mutation_std: float = 0.05,
    mutation_prob: float = 0.1,
    elite_ratio: float = 0.2,
    rl_injection_interval: int = 10,
    crossover_prob: float = 0.5,
    crossover_mode: str = "distillation",
    frac_frames_train: float = 1.0,
    eval_trials: int = 1,
    warmup_steps: int = 1000,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    exploration_noise_std: float = 0.1,
    gradient_steps: int = 100,
    omega: float = 0.3,
    k: int = 5,
    logger: WandbLogger | None = None,
    surrogate_mode: SurrogateMode = SurrogateMode.RANDOM,
    k_ensembles: int = 5,
    beta: float = 1.0,
    debug: bool = False,
    grad_clip_norm: float = 1.0,
    dropout_p: float = 0.2,
    mc_samples: int = 20,
    epsilon: float = 0.2,
    mutation_fraction: float = 0.1,
    mad_k: float = 2.483,
    beta_lr: float = 1e-3,
    policy_delay: int = 2,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    gate_mode: str = "topk",
    rho: float = 0.10,
    rho_min: float = 0.05,
    rho_max: float = 0.5,
    rho_eta: float = 4,
    e_star: float = 0.25,
    e_hat_window: int = 10,
    fitness_norm: str = "tanh",
    h_chunk: int = 25,
    h_alloc: str = "adaptive",
    p_stop: float = 0.05,
    env_id: str = "",
    seed: int = 0,
) -> float:

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    population = [
        Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            action_limit=action_limit,
            activation="tanh",
        ).to(device)
        for _ in range(population_size)
    ]

    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        action_limit=action_limit,
        activation="tanh",
    ).to(device)

    target_actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        action_limit=action_limit,
        activation="tanh",
    ).to(device)

    target_actor.load_state_dict(actor.state_dict())

    critic = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
        activation="elu",
    ).to(device)

    target_critic = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
        activation="elu",
    ).to(device)

    target_critic.load_state_dict(critic.state_dict())

    # TODO: consider passing |Q1 - Q2| disagreement as epistemic uncertainty signal to SurrogateController
    critic_2 = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
        activation="elu",
    ).to(device)
    critic_2_target = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
        activation="elu",
    ).to(device)
    critic_2_target.load_state_dict(critic_2.state_dict())
    critic_2_optimizer = torch.optim.Adam(
        critic_2.parameters(), lr=critic_lr, weight_decay=1e-4
    )

    if surrogate_mode == SurrogateMode.ENSEMBLE:
        critic = EnsembleModule(
            ensemble_size=k_ensembles,
            critic=critic,
            rng=rng,
        ).to(device)

        target_critic = EnsembleModule(
            ensemble_size=k_ensembles,
            critic=target_critic,
            rng=rng,
        ).to(device)

        target_critic.load_state_dict(critic.state_dict())

    if surrogate_mode == SurrogateMode.EVIDENTIAL:
        critic = EvidentialCritic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)

        target_critic = EvidentialCritic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)

        target_critic.load_state_dict(critic.state_dict())

    replay_buffer = Buffer(
        capacity=buffer_size,
        batch_size=batch_size,
        rng=rng,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=critic_lr, weight_decay=1e-4
    )

    evolution_module = EvolutionModule(
        obs_size=state_dim,
        act_size=action_dim,
        net=critic,
        device=device,
    )

    surrogate_controller = SurrogateController(
        evolution_module=evolution_module,
        critic=critic,
        replay_buffer=replay_buffer,
        device=device,
        omega=omega,
        rng=rng,
        k=k,
        surrogate_mode=surrogate_mode,
        beta=beta,
        dropout_p=dropout_p,
        mc_samples=mc_samples,
        epsilon=epsilon,
        crossover_prob=crossover_prob,
        crossover_mode=crossover_mode,
        mad_k=mad_k,
        beta_lr=beta_lr,
        debug=debug,
        gate_mode=gate_mode,
        rho=rho,
        rho_min=rho_min,
        rho_max=rho_max,
        rho_eta=rho_eta,
        e_star=e_star,
        e_hat_window=e_hat_window,
        fitness_norm=fitness_norm,
        gamma=gamma,
        horizon=env.spec.max_episode_steps if env.spec is not None else None,
        h_chunk=h_chunk,
        h_alloc=h_alloc,
        p_stop=p_stop,
    )

    total_steps = warmup(env, replay_buffer, warmup_steps=warmup_steps)

    generation = 0
    recent_rewards = deque(maxlen=100)
    checkpoint_steps = [int(n_steps * f) for f in CHECKPOINT_FRACTIONS]
    next_checkpoint_idx = 0
    env_slug = env_id.replace("/", "_").replace(":", "_") if env_id else "env"

    while total_steps < n_steps:
        generation += 1

        population, fitnesses, evo_steps, used_real_eval = (
            surrogate_controller.generation_based_control(
                population=population,
                env=env,
                evaluate_episodes=eval_trials,
                mutation_std=mutation_std,
                mutation_prob=mutation_prob,
                elite_ratio=elite_ratio,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                mutation_fraction=mutation_fraction,
                actor=actor,
                action_limit=action_limit,
            )
        )
        elite_indices = surrogate_controller.last_elite_indices
        unselect_indices = surrogate_controller.last_unselect_indices

        total_steps += evo_steps

        is_uncertainty_mode = surrogate_mode in (
            SurrogateMode.DROPOUT,
            SurrogateMode.ENSEMBLE,
            SurrogateMode.EVIDENTIAL,
        )

        for fitness in fitnesses:
            recent_rewards.append(fitness)

        best_population_index = int(np.argmax(fitnesses)) if fitnesses else 0
        best_population_member = (
            population[best_population_index] if fitnesses else actor
        )

        eval_reward = evaluate_policy(
            policy=best_population_member,
            env=eval_env,
            device=device,
            episodes=5,
            noise_std=0.0,
        )

        eval_reward_rl_actor = evaluate_policy(
            policy=actor,
            env=eval_env,
            device=device,
            episodes=5,
            noise_std=0.0,
        )

        recent_rewards.append(eval_reward)

        _, rl_steps = rollout_policy(
            policy=actor,
            env=env,
            device=device,
            replay_buffer=replay_buffer,
            episodes=1,
            noise_std=exploration_noise_std,
        )
        total_steps += rl_steps

        critic_loss = 0.0
        actor_loss = 0.0
        if len(replay_buffer) >= batch_size:
            generation_steps = evo_steps + rl_steps
            num_updates = (
                int(generation_steps * frac_frames_train)
                if frac_frames_train > 0.0
                else gradient_steps
            )
            for update_idx in range(num_updates):
                critic_loss = td3_train_critics(
                    actor_target=target_actor,
                    critic_1=critic,
                    critic_2=critic_2,
                    critic_1_target=target_critic,
                    critic_2_target=critic_2_target,
                    critic_1_optimizer=critic_optimizer,
                    critic_2_optimizer=critic_2_optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    policy_noise=policy_noise,
                    noise_clip=noise_clip,
                    action_limit=action_limit,
                    device=device,
                    grad_clip_norm=grad_clip_norm,
                )
                if update_idx % policy_delay == 0:
                    actor_loss = td3_update_actor(
                        actor=actor,
                        critic_1=critic,
                        actor_optimizer=actor_optimizer,
                        replay_buffer=replay_buffer,
                        batch_size=batch_size,
                        device=device,
                        grad_clip_norm=grad_clip_norm,
                    )
                    soft_update(target_critic, critic, tau=tau)
                    soft_update(critic_2_target, critic_2, tau=tau)
                    soft_update(target_actor, actor, tau=tau)

        while (
            surrogate_mode == SurrogateMode.ENSEMBLE
            and next_checkpoint_idx < len(checkpoint_steps)
            and total_steps >= checkpoint_steps[next_checkpoint_idx]
        ):
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            step_label = checkpoint_steps[next_checkpoint_idx]
            ckpt_batch = replay_buffer.sample(batch_size=min(5000, len(replay_buffer)))
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "critic_2": critic_2.state_dict(),
                    "population": [p.state_dict() for p in population],
                    "obs_batch": ckpt_batch["state"].cpu().numpy(),
                    "act_batch": ckpt_batch["action"].cpu().numpy(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "action_limit": action_limit,
                    "actor_hidden_dim": actor_hidden_dim,
                    "k_ensembles": k_ensembles,
                    "seed": seed,
                    "env_id": env_id,
                    "total_steps": total_steps,
                },
                checkpoint_dir / f"ckpt_{env_slug}_seed{seed}_{step_label}.pt",
            )
            next_checkpoint_idx += 1

        behavioural_distances: list[float] = []
        if population and len(replay_buffer) > 0:
            if surrogate_controller.last_obs_batch is not None:
                distance_obs = torch.as_tensor(
                    surrogate_controller.last_obs_batch,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                distance_obs = replay_buffer.sample(
                    batch_size=min(k, len(replay_buffer))
                )["state"].to(device)

            behavioural_distances = [
                behavioural_distance(policy, actor, distance_obs, action_limit)
                for policy in population
            ]

        behavioral_distance_mean = (
            float(np.mean(behavioural_distances)) if behavioural_distances else 0.0
        )
        d_cv = (
            float(np.std(behavioural_distances)) / behavioral_distance_mean
            if behavioural_distances and behavioral_distance_mean > 1e-12
            else 0.0
        )

        if used_real_eval and generation % rl_injection_interval == 0:
            evolution_module.sync_rl_to_pop(
                actor, population, fitnesses, elite_indices, unselect_indices
            )

        avg_fitness = np.mean(fitnesses) if fitnesses else 0.0
        best_fitness = max(fitnesses) if fitnesses else 0.0
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

        if (generation % 10 == 0 or total_steps >= n_steps) and debug:
            print_sc_erl_debug_summary(
                generation=generation,
                total_steps=total_steps,
                avg_fitness=avg_fitness,
                best_fitness=best_fitness,
                avg_reward=avg_reward,
                eval_reward=eval_reward,
                evo_steps=evo_steps,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                uncertainty_mean=(
                    surrogate_controller.last_uncertainty_mean
                    if is_uncertainty_mode
                    else None
                ),
                uncertainty_max=(
                    surrogate_controller.last_uncertainty_max
                    if is_uncertainty_mode
                    else None
                ),
                uncertainty_threshold=(
                    surrogate_controller.last_uncertainty_threshold
                    if is_uncertainty_mode
                    else None
                ),
                surrogate_mode=surrogate_mode.name.lower(),
                raw_sigma_mean=(
                    surrogate_controller.last_raw_sigma_mean
                    if is_uncertainty_mode
                    else None
                ),
                raw_sigma_max=(
                    surrogate_controller.last_raw_sigma_max
                    if is_uncertainty_mode
                    else None
                ),
            )

        if logger is not None:
            metrics = {
                "generation": generation,
                "total_steps": total_steps,
                "avg_population_fitness": avg_fitness,
                "best_population_fitness": best_fitness,
                "avg_recent_reward": avg_reward,
                "eval_reward": eval_reward,
                "eval_reward_best_pop": eval_reward,
                "eval_reward_rl_actor": eval_reward_rl_actor,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "surrogate_used": surrogate_controller.mode == "surrogate",
                "n_real": surrogate_controller.last_n_real,
                "behavioral_distance_mean": behavioral_distance_mean,
                "d_cv": d_cv,
            }
            metrics.update(
                build_surrogate_metrics(
                    surrogate_controller,
                    include_uncertainty=is_uncertainty_mode,
                )
            )

            metrics.update({"surrogate_mode": surrogate_mode.value})

            logger.log(
                metrics,
                step=total_steps,
            )

    return float(eval_reward)
