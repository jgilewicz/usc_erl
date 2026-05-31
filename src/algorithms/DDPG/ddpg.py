import torch
import numpy as np
import gymnasium as gym
from collections import deque

from common.modules import Actor, Critic
from common.reply_buffer import Buffer, Transition
from common.wandb_logger import WandbLogger
from common.utils import (
    evaluate_policy,
    print_ddpg_debug_summary,
    train_actor_step,
    train_critic_step,
    warmup,
)


def ddpg_select_action(
    actor: Actor,
    state: np.ndarray,
    device: torch.device,
    action_limit: float,
    noise_std: float,
) -> np.ndarray:
    actor.eval()
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = actor(state_t).squeeze(0).cpu().numpy()
    noise = noise_std * np.random.randn(*action.shape)
    return np.clip(action + noise, -action_limit, action_limit)


def DDPG(
    buffer_size: int,
    rng: np.random.Generator,
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
    actor_hidden_dim: int = 256,
    critic_hidden_dim: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
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

    critic = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=critic_hidden_dim,
        dropout=0.0,
    ).to(device)

    critic_target = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=critic_hidden_dim,
        dropout=0.0,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

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
    critic_loss = 0.0

    for total_steps in range(1, n_steps + 1):
        if total_steps < warmup_steps:
            action = env.action_space.sample()
        else:
            action = ddpg_select_action(
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

        actor.train()
        critic.train()

        critic_loss = train_critic_step(
            target_actor=actor_target,
            critic=critic,
            target_critic=critic_target,
            critic_optimizer=critic_optimizer,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            grad_clip_norm=grad_clip_norm,
        )

        actor_loss = train_actor_step(
            actor=actor,
            target_actor=actor_target,
            critic=critic,
            actor_optimizer=actor_optimizer,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            tau=tau,
            grad_clip_norm=grad_clip_norm,
        )

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

            if debug:
                print_ddpg_debug_summary(
                    total_steps=total_steps,
                    avg_reward=avg_reward,
                    eval_reward=eval_reward,
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                )

    return float(eval_reward)
