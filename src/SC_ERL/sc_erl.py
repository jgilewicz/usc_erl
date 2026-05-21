from collections import deque

import gymnasium as gym
import numpy as np
import torch

from common.ensemble_module import EnsembleModule
from common.evolution_module import EvolutionModule
from common.modules import Actor, Critic
from common.reply_buffer import Buffer
from common.surrogate_controller import SurrogateController, SurrogateMode
from common.utils import (
    evaluate_policy,
    print_sc_erl_debug_summary,
    rollout_policy,
    train_actor_step,
    train_critic_step,
    warmup,
)
from common.wandb_logger import WandbLogger


def SC_ERL(
    population_size: int,
    buffer_size: int,
    rng: np.random.Generator,
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    hidden_dim: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    mutation_std: float = 0.05,
    mutation_prob: float = 0.1,
    elite_ratio: float = 0.2,
    rl_injection_interval: int = 10,
    warmup_steps: int = 1000,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    exploration_noise_std: float = 0.1,
    gradient_steps: int = 100,
    omega: float = 0.3,
    k: int = 5,
    logger: WandbLogger | None = None,
    surrogate_mode: SurrogateMode = SurrogateMode.RANDOM,
    ensemble_size: int = 5,
    debug: bool = False,
) -> None:

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    population = [
        Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_limit=action_limit,
        ).to(device)
        for _ in range(population_size)
    ]

    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        action_limit=action_limit,
    ).to(device)

    target_actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        action_limit=action_limit,
    ).to(device)

    target_actor.load_state_dict(actor.state_dict())

    critic = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
    ).to(device)

    target_critic = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
    ).to(device)

    target_critic.load_state_dict(critic.state_dict())

    if surrogate_mode == SurrogateMode.ENSEMBLE:
        critic = EnsembleModule(
            ensemble_size=ensemble_size,
            critic=critic,
            rng=rng,
        ).to(device)

        target_critic = EnsembleModule(
            ensemble_size=ensemble_size,
            critic=target_critic,
            rng=rng,
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
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

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
    )

    total_steps = warmup(env, replay_buffer, warmup_steps=warmup_steps)

    generation = 0
    recent_rewards = deque(maxlen=100)

    while total_steps < n_steps:
        generation += 1

        population, fitnesses, evo_steps = (
            surrogate_controller.generation_based_control(
                population=population,
                env=env,
                evaluate_episodes=5,
                mutation_std=mutation_std,
                mutation_prob=mutation_prob,
                elite_ratio=elite_ratio,
            )
        )

        total_steps += evo_steps

        for fitness in fitnesses:
            recent_rewards.append(fitness)

        _, rl_steps = rollout_policy(
            policy=actor,
            env=env,
            device=device,
            replay_buffer=replay_buffer,
            episodes=1,
            noise_std=exploration_noise_std,
        )
        total_steps += rl_steps

        rl_reward = evaluate_policy(
            policy=actor,
            env=env,
            device=device,
            episodes=5,
            noise_std=0.0,
        )

        recent_rewards.append(rl_reward)

        critic_loss = 0.0
        actor_loss = 0.0

        if len(replay_buffer) >= batch_size:
            for _ in range(gradient_steps):
                critic_loss = train_critic_step(
                    target_actor=target_actor,
                    critic=critic,
                    target_critic=target_critic,
                    critic_optimizer=critic_optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    tau=tau,
                )

                actor_loss = train_actor_step(
                    actor=actor,
                    target_actor=target_actor,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    tau=tau,
                )

            # Once every few generations, inject the weakest actor into the population
            if generation % rl_injection_interval == 0:
                eval_fitnesses = [
                    evaluate_policy(ind, eval_env, device=device, episodes=5)
                    for ind in population
                ]

                weakest_idx = int(np.argmin(eval_fitnesses))
                population[weakest_idx].load_state_dict(actor.state_dict())

            avg_fitness = np.mean(fitnesses) if fitnesses else 0.0
            best_fitness = max(fitnesses) if fitnesses else 0.0
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

            if generation % 10 == 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                best_fitness = max(fitnesses) if fitnesses else 0.0
                avg_fitness = np.mean(fitnesses) if fitnesses else 0.0

                if debug:
                    print_sc_erl_debug_summary(
                        generation=generation,
                        total_steps=total_steps,
                        avg_fitness=avg_fitness,
                        best_fitness=best_fitness,
                        avg_reward=avg_reward,
                        rl_reward=rl_reward,
                        evo_steps=evo_steps,
                        actor_loss=actor_loss,
                        critic_loss=critic_loss,
                        uncertainty_mean=(
                            surrogate_controller.last_uncertainty_mean
                            if surrogate_mode
                            in (SurrogateMode.DROPOUT, SurrogateMode.ENSEMBLE)
                            else None
                        ),
                        uncertainty_max=(
                            surrogate_controller.last_uncertainty_max
                            if surrogate_mode
                            in (SurrogateMode.DROPOUT, SurrogateMode.ENSEMBLE)
                            else None
                        ),
                        uncertainty_threshold=(
                            surrogate_controller.last_uncertainty_threshold
                            if surrogate_mode
                            in (SurrogateMode.DROPOUT, SurrogateMode.ENSEMBLE)
                            else None
                        ),
                        surrogate_mode=surrogate_mode.name.lower(),
                    )

                if logger is not None:
                    metrics = {
                        "generation": generation,
                        "total_steps": total_steps,
                        "evo_steps": evo_steps,
                        "avg_population_fitness": avg_fitness,
                        "best_population_fitness": best_fitness,
                        "avg_recent_reward": avg_reward,
                        "rl_reward": rl_reward,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "surrogate_used": surrogate_controller.mode == "surrogate",
                    }
                    if surrogate_mode in (
                        SurrogateMode.DROPOUT,
                        SurrogateMode.ENSEMBLE,
                    ):
                        metrics.update(
                            {
                                "uncertainty_mean": surrogate_controller.last_uncertainty_mean,
                                "uncertainty_max": surrogate_controller.last_uncertainty_max,
                                "uncertainty_threshold": surrogate_controller.last_uncertainty_threshold,
                            }
                        )

                    metrics.update(
                        {
                            "surrogate_mode": surrogate_mode.value,
                            "omega": omega,
                            "k": k,
                        }
                    )

                    logger.log(
                        metrics,
                        step=generation,
                    )
