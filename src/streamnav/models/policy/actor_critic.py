from torch import nn


class NavigationPooling(nn.Module):
    def forward(self, hidden_states):
        # encode_rgb always appends exactly one learned NAV token.
        return hidden_states[:, -1]


class NavigationActorCritic(nn.Module):
    def __init__(self, hidden_dim=1024, value_hidden_dim=512):
        super().__init__()
        self.actor = nn.Linear(hidden_dim, 4)
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, value_hidden_dim),
            nn.GELU(),
            nn.Linear(value_hidden_dim, 1),
        )
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, hidden):
        return self.actor(hidden), self.critic(hidden).squeeze(-1)
