import torch
import torch.nn as nn
import copy
from common.reply_buffer import Buffer
from common.modules import Critic, Actor


class MCDropout:
    @staticmethod
    def _set_dropout_probability(module: nn.Module, p: float) -> None:
        for submodule in module.modules():
            if isinstance(submodule, nn.Dropout):
                submodule.p = p

    @staticmethod
    def make_dropout_critic(
        critic: Critic,
        dropout_p: float = 0.2,
        device: str | torch.device = "cpu",
    ) -> Critic:
        dropout_critic = copy.deepcopy(critic).to(device)
        MCDropout._set_dropout_probability(dropout_critic, dropout_p)
        return dropout_critic

    @staticmethod
    def _mc_dropout_fitness(
        critic: Critic,
        policy: Actor,
        obs: torch.Tensor,
        device: str | torch.device = "cpu",
        T: int = 20,
    ) -> tuple[float, float]:
        policy = policy.to(device)
        obs = obs.to(device)

        critic_was_training = critic.training
        policy_was_training = policy.training

        critic.train()
        policy.eval()

        try:
            q_samples = []

            with torch.no_grad():
                actions = policy(obs)

                for _ in range(T):
                    q = critic(obs, actions).squeeze(-1)
                    q_samples.append(q)

            q_samples = torch.stack(q_samples, dim=0)

            per_sample_mean = q_samples.mean(dim=0)
            per_sample_std = q_samples.std(dim=0)

            fitness = per_sample_mean.mean().item()
            uncertainty = per_sample_std.mean().item()
            return fitness, uncertainty
        finally:
            critic.train(critic_was_training)
            policy.train(policy_was_training)

    @staticmethod
    def fitness_evaluation_mc_dropout(
        critic: Critic,
        population: list[Actor],
        replay_buffer: Buffer,
        k: int,
        device: str | torch.device = "cpu",
        T: int = 20,
        dropout_p: float = 0.2,
    ) -> tuple[list[float], list[float]]:
        k = min(k, len(replay_buffer))
        batch = replay_buffer.sample_latest(batch_size=k)
        obs = batch["state"].to(device)

        dropout_critic = MCDropout.make_dropout_critic(
            critic=critic, dropout_p=dropout_p, device=device
        )

        fitness = []
        uncertainty = []

        for policy in population:
            f, u = MCDropout._mc_dropout_fitness(
                dropout_critic, policy, obs, device, T=T
            )
            fitness.append(f)
            uncertainty.append(u)

        return fitness, uncertainty
