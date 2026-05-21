import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from common.modules import StochasticActor, StateCritic
from common.reply_buffer import RolloutBuffer, PPOTransition
from common.wandb_logger import WandbLogger


def evaluate_ppo_policy(
    policy: StochasticActor,
    env: gym.Env,
    device: torch.device,
    episodes: int = 1,
) -> tuple[float, int]:
    policy.eval()

    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = policy(obs_t)
                action = dist.mean.squeeze(0).cpu().numpy()  # deterministic evaluation using mean

            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            episode_reward += reward
            total_steps += 1

        total_reward += episode_reward

    return total_reward / episodes, total_steps


def print_ppo_debug_summary(
    total_steps: int,
    avg_reward: float,
    eval_reward: float,
    actor_loss: float,
    critic_loss: float,
    entropy: float,
) -> None:
    print(f"[PPO] Steps {total_steps:,}")
    print(f"  Rewards     recent avg: {avg_reward:8.2f} | eval: {eval_reward:8.2f}")
    print(f"  Optimization actor loss: {actor_loss:8.4f} | critic loss: {critic_loss:8.4f} | entropy: {entropy:8.4f}")
    print()


def PPO(
    env: gym.Env,
    eval_env: gym.Env,
    n_steps: int,
    rollout_steps: int = 2048,
    batch_size: int = 64,
    ppo_epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    hidden_dim: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_param: float = 0.2,
    entropy_coef: float = 0.0,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    evaluate_episodes: int = 5,
    eval_interval: int = 5000,
    logger: WandbLogger | None = None,
    debug: bool = False,
) -> None:
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = StochasticActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    critic = StateCritic(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr, eps=1e-5)

    rollout_buffer = RolloutBuffer(capacity=rollout_steps, device=device)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    
    total_steps = 0
    updates = 0
    
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_entropy = 0.0

    while total_steps < n_steps:
        actor.eval()
        critic.eval()
        
        # Collect rollout
        for _ in range(rollout_steps):
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                dist = actor(state_t)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = critic(state_t).squeeze(-1)
                
            action_np = action.cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            rollout_buffer.add(
                PPOTransition(
                    state=state,
                    action=action_np,
                    log_prob=log_prob.item(),
                    reward=reward,
                    value=value.item(),
                    done=done,
                )
            )

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                state, _ = env.reset()
                episode_reward = 0.0

            # Evaluation
            if total_steps % eval_interval == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
                eval_reward, _ = evaluate_ppo_policy(actor, eval_env, device, episodes=evaluate_episodes)

                if logger is not None:
                    logger.log(
                        {
                            "total_steps": total_steps,
                            "avg_reward": avg_reward,
                            "eval_reward": eval_reward,
                        },
                        step=total_steps,
                    )
                
                if debug:
                    print_ppo_debug_summary(
                        total_steps=total_steps,
                        avg_reward=avg_reward,
                        eval_reward=eval_reward,
                        actor_loss=last_actor_loss,
                        critic_loss=last_critic_loss,
                        entropy=last_entropy,
                    )
                    
            if total_steps >= n_steps:
                break

        # Compute GAE
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            last_value = critic(state_t).squeeze(-1).item()
        
        rollout_buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

        # Optimize
        actor.train()
        critic.train()
        
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_entropy = 0.0
        
        num_updates = 0

        for _ in range(ppo_epochs):
            for batch in rollout_buffer.get_generator(batch_size):
                b_states = batch["state"]
                b_actions = batch["action"]
                b_log_probs = batch["log_prob"]
                b_returns = batch["return"]
                b_advantages = batch["advantage"]

                dist = actor(b_states)
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(new_log_probs - b_log_probs)
                
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

                values = critic(b_states).squeeze(-1)
                critic_loss = F.mse_loss(values, b_returns)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()
                
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy.item()
                num_updates += 1

        rollout_buffer.reset()
        
        if num_updates > 0:
            last_actor_loss = epoch_actor_loss / num_updates
            last_critic_loss = epoch_critic_loss / num_updates
            last_entropy = epoch_entropy / num_updates

        if logger is not None and num_updates > 0:
            logger.log(
                {
                    "actor_loss": last_actor_loss,
                    "critic_loss": last_critic_loss,
                    "entropy": last_entropy,
                },
                step=total_steps,
            )
