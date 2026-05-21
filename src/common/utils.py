from torch import nn
import torch
import torch.nn.functional as F
from common.reply_buffer import Buffer, Transition
import gymnasium as gym
import numpy as np
from common.modules import Actor, Critic


def format_steps(value: int) -> str:
    return f"{value:,}"


def print_td3_debug_summary(
    total_steps: int,
    avg_reward: float,
    eval_reward: float,
    actor_loss: float,
    critic_loss: float,
) -> None:
    print(f"[TD3] Steps {format_steps(total_steps)}")
    print(
        f"  Rewards     recent avg: {avg_reward:8.2f} | eval: {eval_reward:8.2f}"
    )
    print(
        f"  Optimization actor loss: {actor_loss:8.4f} | critic loss: {critic_loss:8.4f}"
    )
    print()


def print_erl_debug_summary(
    generation: int,
    total_steps: int,
    avg_fitness: float,
    best_fitness: float,
    avg_reward: float,
    rl_reward: float,
    actor_loss: float,
    critic_loss: float,
) -> None:
    print(f"[ERL] Generation {generation} | Steps {format_steps(total_steps)}")
    print(
        f"  Population  avg: {avg_fitness:8.2f} | best: {best_fitness:8.2f} | "
        f"recent avg reward: {avg_reward:8.2f}"
    )
    print(
        f"  RL policy       reward: {rl_reward:8.2f} | actor loss: {actor_loss:8.4f} | "
        f"critic loss: {critic_loss:8.4f}"
    )
    print()


def print_sc_erl_debug_summary(
    generation: int,
    total_steps: int,
    avg_fitness: float,
    best_fitness: float,
    avg_reward: float,
    rl_reward: float,
    evo_steps: int,
    actor_loss: float,
    critic_loss: float,
    uncertainty_mean: float | None = None,
    uncertainty_max: float | None = None,
    uncertainty_threshold: float | None = None,
    surrogate_mode: str | None = None,
) -> None:
    print(f"[SC-ERL] Generation {generation} | Steps {format_steps(total_steps)}")
    print(
        f"  Population  avg: {avg_fitness:8.2f} | best: {best_fitness:8.2f} | "
        f"recent avg reward: {avg_reward:8.2f}"
    )
    print(
        f"  RL policy       reward: {rl_reward:8.2f} | actor loss: {actor_loss:8.4f} | "
        f"critic loss: {critic_loss:8.4f}"
    )
    print(f"  Evolution       steps: {format_steps(evo_steps)}")
    if (
        surrogate_mode in ("dropout", "ensemble")
        and uncertainty_mean is not None
        and uncertainty_max is not None
        and uncertainty_threshold is not None
    ):
        print(
            f"  Uncertainty     mean: {uncertainty_mean:8.4f} | max: {uncertainty_max:8.4f} | "
            f"threshold: {uncertainty_threshold:8.4f}"
        )
    print()


def get_flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in module.parameters()])


def set_flat_params(
    module: nn.Module, flat_params: torch.Tensor, device: torch.device = "cpu"
) -> None:
    flat_params = flat_params.to(device)
    offset = 0
    for param in module.parameters():
        elements = param.numel()
        param.data.copy_(flat_params[offset : offset + elements].view_as(param))
        offset += elements


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def train_critic_step(
    target_actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    critic_optimizer: torch.optim.Optimizer,
    replay_buffer: Buffer,
    batch_size: int,
    gamma: float = 0.99,
    tau: float = 0.005,
) -> float:
    batch = replay_buffer.sample(batch_size=batch_size)

    with torch.no_grad():
        next_action = target_actor(batch["next_state"])
        if hasattr(target_critic, "compute_loss"):
            next_q_mean, _ = target_critic(batch["next_state"], next_action)
            next_q = next_q_mean
        else:
            next_q = target_critic(batch["next_state"], next_action)

        target_q = batch["reward"] + (1.0 - batch["done"]) * gamma * next_q

    if hasattr(critic, "compute_loss"):
        critic_loss = critic.compute_loss(batch["state"], batch["action"], target_q)
    else:
        critic_loss = torch.nn.MSELoss()(critic(batch["state"], batch["action"]), target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    soft_update(target_critic, critic, tau=tau)

    return critic_loss.item()


def train_actor_step(
    actor: nn.Module,
    target_actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    replay_buffer: Buffer,
    batch_size: int,
    tau: float = 0.005,
) -> float:
    batch = replay_buffer.sample(batch_size=batch_size)

    q_values = critic(batch["state"], actor(batch["state"]))
    if isinstance(q_values, tuple):
        q_values = q_values[0] # use mean_q for ensemble

    actor_loss = -q_values.mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(target_actor, actor, tau=tau)

    return actor_loss.item()


def warmup(env: gym.Env, replay_buffer: Buffer, warmup_steps: int) -> int:
    state, _ = env.reset()
    total_steps = 0

    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )
        )

        if done:
            state, _ = env.reset()
        else:
            state = next_obs
        total_steps += 1

    return total_steps

def rollout_policy(
    policy: Actor,
    env: gym.Env,
    device: torch.device,
    replay_buffer: Buffer,
    episodes: int = 1,
    noise_std: float = 0.0,
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
                action = policy(obs_t).squeeze(0).cpu().numpy()
            if noise_std > 0.0:
                action += noise_std * np.random.randn(*action.shape)

            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(
                Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                )
            )

            obs = next_obs
            episode_reward += reward
            total_steps += 1

        total_reward += episode_reward

    return total_reward / episodes, total_steps



def evaluate_policy(
    policy: Actor,
    env: gym.Env,
    device: torch.device,
    episodes: int = 1,
    noise_std: float = 0.0,
) -> float:
    policy.eval()

    total_reward = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_t).squeeze(0).cpu().numpy()
            if noise_std > 0.0:
                action += noise_std * np.random.randn(*action.shape)

            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / episodes


def fintess_policy_evaluation(
    self,
    population: list[nn.Module],
    replay_buffer: Buffer,
    critic: nn.Module,
    k: int = 5,
) -> list[float]:

    def _fitness_one_policy(
        policy: nn.Module, obs: torch.Tensor, critic: nn.Module
    ) -> float:
        policy = policy.to(self.device)
        policy.eval()

        obs_t = obs.to(self.device)
        critic.eval()

        with torch.no_grad():
            actions = policy(obs_t)
            q_values = critic(obs_t, actions).squeeze(-1)

        return q_values.mean().item()

    k = min(k, len(replay_buffer))
    batch = replay_buffer.sample_latest(batch_size=k)
    obs = batch["state"].to(self.device)

    fitness = []

    for policy in population:
        fitness.append(_fitness_one_policy(policy, obs, critic))

    return fitness


def td3_select_action(
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


def td3_train_critics(
    actor_target: Actor,
    critic_1: Critic,
    critic_2: Critic,
    critic_1_target: Critic,
    critic_2_target: Critic,
    critic_1_optimizer: torch.optim.Optimizer,
    critic_2_optimizer: torch.optim.Optimizer,
    replay_buffer: Buffer,
    batch_size: int,
    gamma: float,
    policy_noise: float,
    noise_clip: float,
    action_limit: float,
    device: torch.device,
) -> float:
    batch = replay_buffer.sample(batch_size)
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    reward = batch["reward"].to(device)
    next_state = batch["next_state"].to(device)
    done = batch["done"].to(device)

    with torch.no_grad():
        noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
        next_action = (actor_target(next_state) + noise).clamp(
            -action_limit, action_limit
        )

        target_q1 = critic_1_target(next_state, next_action)
        target_q2 = critic_2_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)

        target = reward + (1.0 - done) * gamma * target_q

    current_q1 = critic_1(state, action)
    current_q2 = critic_2(state, action)
    critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

    critic_1_optimizer.zero_grad()
    critic_2_optimizer.zero_grad()
    critic_loss.backward()
    critic_1_optimizer.step()
    critic_2_optimizer.step()

    return critic_loss.item()


def td3_update_actor(
    actor: Actor,
    critic_1: Critic,
    actor_optimizer: torch.optim.Optimizer,
    replay_buffer: Buffer,
    batch_size: int,
    device: torch.device,
) -> float:
    batch = replay_buffer.sample(batch_size)
    state = batch["state"].to(device)

    actor_loss = -critic_1(state, actor(state)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item()


def surrogate_fitness(
    population: list[nn.Module],
    critic: nn.Module,
    replay_buffer: Buffer,
    device: torch.device,
    k: int = 5,
) -> list[float]:
    k = min(k, len(replay_buffer))
    batch = replay_buffer.sample_latest(batch_size=k)
    obs = batch["state"].to(device)

    critic_was_training = critic.training
    critic.eval()

    fitnesses = []
    try:
        for policy in population:
            policy = policy.to(device)
            policy_was_training = policy.training
            try:
                policy.eval()

                with torch.no_grad():
                    actions = policy(obs)
                    q_values = critic(obs, actions).squeeze(-1)
                    fitnesses.append(q_values.mean().item())
            finally:
                policy.train(policy_was_training)
    finally:
        critic.train(critic_was_training)

    return fitnesses
