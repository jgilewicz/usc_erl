from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch import nn

from common.reply_buffer import Buffer, Transition
from modules.deep_modules import Actor, Critic, EvidentialCritic
from modules.ensemble_module import EnsembleModule

if TYPE_CHECKING:
    from common.surrogate_controller import SurrogateController


def _unwrap_q(output) -> torch.Tensor:
    # EnsembleModule returns (mean_q, std_q); plain Critic returns a tensor
    return output[0] if isinstance(output, tuple) else output


def format_steps(value: int) -> str:
    return f"{value:,}"


def print_erl_debug_summary(
    generation: int,
    total_steps: int,
    avg_fitness: float,
    best_fitness: float,
    avg_reward: float,
    eval_reward: float,
    actor_loss: float,
    critic_loss: float,
) -> None:
    print(f"[ERL] Generation {generation} | Steps {format_steps(total_steps)}")
    print(
        f"  Population  avg: {avg_fitness:8.2f} | best: {best_fitness:8.2f} | "
        f"recent avg reward: {avg_reward:8.2f}"
    )
    print(
        f"  Best pop reward  reward: {eval_reward:8.2f} | actor loss: {
            actor_loss:8.4f} | "
        f"critic loss: {critic_loss:8.4f}"
    )
    print()


def print_sc_erl_debug_summary(
    generation: int,
    total_steps: int,
    avg_fitness: float,
    best_fitness: float,
    avg_reward: float,
    eval_reward: float,
    evo_steps: int,
    actor_loss: float,
    critic_loss: float,
    uncertainty_mean: float | None = None,
    uncertainty_max: float | None = None,
    uncertainty_threshold: float | None = None,
    surrogate_mode: str | None = None,
    raw_sigma_mean: float | None = None,
    raw_sigma_max: float | None = None,
) -> None:
    print(f"[SC-ERL] Generation {generation} | Steps {format_steps(total_steps)}")
    print(
        f"  Population  avg: {avg_fitness:8.2f} | best: {best_fitness:8.2f} | "
        f"recent avg reward: {avg_reward:8.2f}"
    )
    print(
        f"  Best pop reward  reward: {eval_reward:8.2f} | actor loss: {
            actor_loss:8.4f} | "
        f"critic loss: {critic_loss:8.4f}"
    )
    print(f"  Evolution       steps: {format_steps(evo_steps)}")
    if (
        surrogate_mode in ("dropout", "ensemble", "evidential")
        and uncertainty_mean is not None
        and uncertainty_max is not None
        and uncertainty_threshold is not None
    ):
        print(
            f"  CV uncertainty  mean: {uncertainty_mean:8.4f} | max: {
                uncertainty_max:8.4f} | "
            f"threshold: {uncertainty_threshold:8.4f}"
        )
        if raw_sigma_mean is not None and raw_sigma_max is not None:
            print(
                f"  Raw sigma       mean: {raw_sigma_mean:8.4f} | max: {
                    raw_sigma_max:8.4f}"
            )
    print()


def get_flat_params(module: nn.Module) -> torch.Tensor:
    params = []
    excluded_params = set()
    for m in module.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters():
                excluded_params.add(p)

    for p in module.parameters():
        if p not in excluded_params:
            params.append(p.detach().view(-1))
    return torch.cat(params)


def set_flat_params(
    module: nn.Module, flat_params: torch.Tensor, device: torch.device = "cpu"
) -> None:
    flat_params = flat_params.to(device)
    offset = 0
    excluded_params = set()
    for m in module.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters():
                excluded_params.add(p)

    for param in module.parameters():
        if param not in excluded_params:
            elements = param.numel()
            param.data.copy_(flat_params[offset : offset + elements].view_as(param))
            offset += elements


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


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
                done=terminated,
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
    store_in_buffer: bool = True,
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

            if store_in_buffer and replay_buffer is not None:
                replay_buffer.add(
                    Transition(
                        state=obs,
                        action=action,
                        reward=reward,
                        next_state=next_obs,
                        done=terminated,
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


def td3_train_critics(
    actor_target: Actor,
    critic_1: Critic | EnsembleModule,
    critic_2: Critic,
    critic_1_target: Critic | EnsembleModule,
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
    grad_clip_norm: float = 1.0,
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

        target_q2 = _unwrap_q(critic_2_target(next_state, next_action))

        target_per_member = None
        if isinstance(critic_1_target, EnsembleModule):
            target_q1_per_member = critic_1_target.forward_per_member(
                next_state, next_action
            )
            target_q1_mean = target_q1_per_member.mean(dim=0)
            target_per_member = reward.unsqueeze(0) + (
                1.0 - done.unsqueeze(0)
            ) * gamma * torch.min(target_q1_per_member, target_q2.unsqueeze(0))
        else:
            target_q1_mean = _unwrap_q(critic_1_target(next_state, next_action))

        target = reward + (1.0 - done) * gamma * torch.min(target_q1_mean, target_q2)

    if isinstance(critic_1, EnsembleModule):
        assert target_per_member is not None, (
            "critic_1 is an EnsembleModule but critic_1_target is not; "
            "both must be EnsembleModule together"
        )
        loss_1 = critic_1.compute_loss(state, action, target_per_member)
    else:
        loss_1 = F.smooth_l1_loss(_unwrap_q(critic_1(state, action)), target)

    loss_2 = F.smooth_l1_loss(_unwrap_q(critic_2(state, action)), target)
    critic_loss = loss_1 + loss_2

    critic_1_optimizer.zero_grad()
    critic_2_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic_1.parameters(), grad_clip_norm)
    torch.nn.utils.clip_grad_norm_(critic_2.parameters(), grad_clip_norm)
    critic_1_optimizer.step()
    critic_2_optimizer.step()

    return critic_loss.item()


def td3_update_actor(
    actor: Actor,
    critic_1: Critic | EnsembleModule,
    actor_optimizer: torch.optim.Optimizer,
    replay_buffer: Buffer,
    batch_size: int,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> float:
    batch = replay_buffer.sample(batch_size)
    state = batch["state"].to(device)

    actor_loss = -_unwrap_q(critic_1(state, actor(state))).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm)
    actor_optimizer.step()

    return actor_loss.item()


def surrogate_fitness(
    population: list[Actor],
    critic: Critic | EnsembleModule | EvidentialCritic,
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
                    q_values = critic(obs, actions)
                    if isinstance(q_values, tuple):
                        q_values = q_values[0]
                    q_values = q_values.squeeze(-1)
                    fitnesses.append(q_values.mean().item())
            finally:
                policy.train(policy_was_training)
    finally:
        critic.train(critic_was_training)

    return fitnesses


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    r = rankdata(scores)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def build_surrogate_metrics(
    surrogate_controller: "SurrogateController", include_uncertainty: bool
) -> dict[str, float]:
    if not include_uncertainty:
        return {}
    metrics: dict[str, float] = {
        "uncertainty_mean": surrogate_controller.last_uncertainty_mean,
        "uncertainty_max": surrogate_controller.last_uncertainty_max,
        "uncertainty_threshold": surrogate_controller.last_uncertainty_threshold,
        "raw_sigma_mean": surrogate_controller.last_raw_sigma_mean,
        "raw_sigma_max": surrogate_controller.last_raw_sigma_max,
        "raw_sigma_cv": surrogate_controller.raw_sigma_cv,
        "rho": surrogate_controller.rho,
        "e_hat_mean": surrogate_controller.e_hat_mean,
    }
    if surrogate_controller.last_gate_quality is not None:
        metrics.update(
            {
                f"gate_{name}": value
                for name, value in surrogate_controller.last_gate_quality.items()
            }
        )
    return metrics


def behavioural_distance(
    policy_j: nn.Module,
    policy_rl: nn.Module,
    obs: torch.Tensor,
    a_max: float,
) -> float:
    with torch.no_grad():
        a_j = policy_j(obs)
        a_rl = policy_rl(obs)

        diff = (a_j - a_rl) / a_max

        per_state = diff.pow(2).mean(dim=1).sqrt()
        return float(per_state.mean())
