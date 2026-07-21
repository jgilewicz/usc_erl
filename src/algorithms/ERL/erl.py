import torch
import numpy as np
import gymnasium as gym
from collections import deque

from modules.deep_modules import Actor, Critic
from common.reply_buffer import Buffer
from modules.evolution_module import EvolutionModule
from common.wandb_logger import WandbLogger
from common.utils import (
    td3_train_critics,
    td3_update_actor,
    soft_update,
    warmup,
    evaluate_policy,
    rollout_policy,
    print_erl_debug_summary,
)


def ERL(
    population_size: int,
    buffer_size: int,
    rng: np.random.Generator,
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
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
    evaluate_episodes: int = 5,
    exploration_noise_std: float = 0.1,
    gradient_steps: int = 100,
    logger: WandbLogger | None = None,
    debug: bool = False,
    grad_clip_norm: float = 1.0,
    mutation_fraction: float = 0.1,
    policy_delay: int = 2,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
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
    critic_2_optimizer = torch.optim.Adam(
        critic_2.parameters(), lr=critic_lr, weight_decay=1e-4
    )

    evolution_module = EvolutionModule(
        obs_size=state_dim,
        act_size=action_dim,
        net=critic,
        device=device,
    )

    total_steps = warmup(env, replay_buffer, warmup_steps=warmup_steps)

    generation = 0
    recent_rewards = deque(maxlen=100)

    while total_steps < n_steps:
        generation += 1

        fitnesses = []
        critic_loss = 0.0
        actor_loss = 0.0
        generation_steps = 0

        for individual in population:
            fitness, steps = rollout_policy(
                policy=individual,
                env=env,
                device=device,
                replay_buffer=replay_buffer,
                episodes=eval_trials,
                noise_std=0.0,
            )

            fitnesses.append(fitness)
            total_steps += steps
            generation_steps += steps
            recent_rewards.append(fitness)

        best_population_index = int(np.argmax(fitnesses)) if fitnesses else 0
        best_population_member = (
            population[best_population_index] if fitnesses else actor
        )

        eval_reward = evaluate_policy(
            policy=best_population_member,
            env=eval_env,
            device=device,
            episodes=evaluate_episodes,
            noise_std=0.0,
        )

        recent_rewards.append(eval_reward)

        population, elite_indices, unselect_indices = evolution_module.evolve(
            population=population,
            fitnesses=fitnesses,
            mutation_std=mutation_std,
            mutation_prob=mutation_prob,
            elite_ratio=elite_ratio,
            crossover_prob=crossover_prob,
            crossover_mode=crossover_mode,
            replay_buffer=replay_buffer,
            mutation_fraction=mutation_fraction,
        )

        _, rl_steps = rollout_policy(
            policy=actor,
            env=env,
            device=device,
            replay_buffer=replay_buffer,
            episodes=1,
            noise_std=exploration_noise_std,
        )
        total_steps += rl_steps
        generation_steps += rl_steps

        if len(replay_buffer) >= batch_size:
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

        if generation % rl_injection_interval == 0:
            evolution_module.sync_rl_to_pop(
                actor, population, fitnesses, elite_indices, unselect_indices
            )

        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        best_fitness = max(fitnesses) if fitnesses else 0.0
        avg_fitness = np.mean(fitnesses) if fitnesses else 0.0

        if generation % 10 == 0 or total_steps >= n_steps:
            if debug:
                print_erl_debug_summary(
                    generation=generation,
                    total_steps=total_steps,
                    avg_fitness=avg_fitness,
                    best_fitness=best_fitness,
                    avg_reward=avg_reward,
                    eval_reward=eval_reward,
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                )

        if logger is not None:
            logger.log(
                {
                    "generation": generation,
                    "total_steps": total_steps,
                    "avg_population_fitness": avg_fitness,
                    "best_population_fitness": best_fitness,
                    "avg_recent_reward": avg_reward,
                    "eval_reward": eval_reward,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                },
                step=total_steps,
            )

    return float(eval_reward)
