import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_limit: float = 1.0,
        activation: str = "relu",
        use_ln: bool = True,
    ):
        super(Actor, self).__init__()

        self.action_limit = action_limit
        act_layer = nn.Tanh() if activation.lower() == "tanh" else nn.ReLU()

        layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim)]
        if use_ln:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_layer)
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_ln:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_layer)
        self.net = nn.Sequential(*layers)

        self.out_layer = nn.Linear(hidden_dim, action_dim)

        nn.init.uniform_(self.out_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out_layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        return torch.tanh(self.out_layer(x)) * self.action_limit


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        hidden_dims: list[int] = [400, 300],
    ) -> None:
        super(Critic, self).__init__()

        act_layer = nn.ELU() if activation.lower() == "elu" else nn.ReLU()
        dim1, dim2 = hidden_dims[0], hidden_dims[1]

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, dim1),
            nn.LayerNorm(dim1),
            act_layer,
        )

        self.net = nn.Sequential(
            nn.Linear(dim1 + action_dim, dim2),
            nn.LayerNorm(dim2),
            act_layer,
            nn.Dropout(p=dropout),
            nn.Linear(dim2, dim2),
            nn.LayerNorm(dim2),
            act_layer,
            nn.Dropout(p=dropout),
            nn.Linear(dim2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_features = self.state_net(state)
        x = torch.cat([state_features, action], dim=-1)
        return self.net(x)


class StochasticActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super(StochasticActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        nn.init.uniform_(self.mu_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu_layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.distributions.Normal:
        x = self.net(state)
        mu = self.mu_layer(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)


class StateCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(StateCritic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class EvidentialModule(nn.Module):
    _EPS: float = 1e-4

    def __init__(self, in_features: int, units: int = 1):
        super(EvidentialModule, self).__init__()
        self.units = units
        self.dense = nn.Linear(in_features, 4 * self.units)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.units, dim=-1)
        # Floor v and beta to avoid zero denominators in epistemic variance
        v = self.evidence(logv) + self._EPS
        alpha = self.evidence(logalpha) + 1.0 + self._EPS
        beta = self.evidence(logbeta) + self._EPS
        # Clamp mu to a safe range — prevents extreme Q-values from producing
        # NaN actor gradients that would destabilise the MuJoCo simulation
        mu = torch.clamp(mu, -1e4, 1e4)
        return mu, v, alpha, beta


class EvidentialCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [400, 300],
    ) -> None:
        super(EvidentialCritic, self).__init__()
        dim1, dim2 = hidden_dims[0], hidden_dims[1]

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, dim1),
            nn.LayerNorm(dim1),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(dim1 + action_dim, dim2),
            nn.LayerNorm(dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim2),
            nn.LayerNorm(dim2),
            nn.ReLU(),
        )

        self.evidential_output = EvidentialModule(in_features=dim2, units=1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        state_features = self.state_net(state)
        x = torch.cat([state_features, action], dim=-1)
        x = self.net(x)

        return self.evidential_output(x)


class AdaptiveBeta(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.log_beta = nn.Parameter(torch.tensor(float(init_value)).log())

    @property
    def beta(self) -> float:
        return float(torch.exp(self.log_beta).item())
