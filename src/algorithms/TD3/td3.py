import torch
import numpy as np
import gymnasium as gym
from collections import deque

from modules.deep_modules import Actor, Critic
from common.reply_buffer import Buffer, Transition
from common.wandb_logger import WandbLogger
from common.utils import (
    evaluate_policy,
    print_td3_debug_summary,
    soft_update,
    td3_select_action,
    td3_train_critics,
    td3_update_actor,
)


def TD3(
    buffer_size: int,
    rng: np.random.Generator,
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    actor_hidden_dim: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    exploration_noise_std: float = 0.1,
    warmup_steps: int = 1000,
    evaluate_episodes: int = 5,
    eval_interval: int = 5000,
    logger: WandbLogger | None = None,
    debug: bool = False,
    grad_clip_norm: float = 1.0,
) -> float:

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        action_limit=action_limit,
    ).to(device)

    actor_target = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        action_limit=action_limit,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic_1 = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
    ).to(device)

    critic_2 = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
    ).to(device)

    critic_1_target = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
    ).to(device)
    critic_1_target.load_state_dict(critic_1.state_dict())

    critic_2_target = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        dropout=0.0,
    ).to(device)
    critic_2_target.load_state_dict(critic_2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=critic_lr)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=critic_lr)

    replay_buffer = Buffer(
        capacity=buffer_size,
        batch_size=batch_size,
        rng=rng,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    episode_rewards = deque(maxlen=100)

    state, _ = env.reset()
    episode_reward = 0.0
    avg_reward = 0.0
    eval_reward = 0.0
    actor_loss = 0.0

    for total_steps in range(1, n_steps + 1):
        if total_steps < warmup_steps:
            action = env.action_space.sample()
        else:
            action = td3_select_action(
                actor, state, device, action_limit, exploration_noise_std
            )

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,
            )
        )

        state = next_state
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0.0

        if len(replay_buffer) < batch_size:
            continue

        critic_loss = td3_train_critics(
            actor_target=actor_target,
            critic_1=critic_1,
            critic_2=critic_2,
            critic_1_target=critic_1_target,
            critic_2_target=critic_2_target,
            critic_1_optimizer=critic_1_optimizer,
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

        if total_steps % policy_delay == 0:
            actor_loss = td3_update_actor(
                actor=actor,
                critic_1=critic_1,
                actor_optimizer=actor_optimizer,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                device=device,
                grad_clip_norm=grad_clip_norm,
            )

            soft_update(critic_1_target, critic_1, tau)
            soft_update(critic_2_target, critic_2, tau)
            soft_update(actor_target, actor, tau)

        if total_steps % eval_interval == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            eval_reward = evaluate_policy(
                actor, eval_env, device, episodes=evaluate_episodes
            )

            if logger is not None:
                logger.log(
                    {
                        "total_steps": total_steps,
                        "avg_reward": avg_reward,
                        "eval_reward": eval_reward,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                    },
                    step=total_steps,
                )

        if debug and total_steps % eval_interval == 0:
            print_td3_debug_summary(
                total_steps=total_steps,
                avg_reward=avg_reward,
                eval_reward=eval_reward,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
            )

    return float(eval_reward)
