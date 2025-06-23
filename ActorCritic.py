import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(config.STATE_SIZE, 512)
        self.fc2 = nn.Linear(512, 512)
        if config.USER_ITEM_SEPARATION:
            self.mean_linear = nn.Linear(512, config.ACTION_SIZE)
            self.log_std_linear = nn.Linear(512, config.ACTION_SIZE)
        else:
            self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

        self.optimiser = optim.Adam(params=self.parameters(), lr=2e-4, weight_decay=1e-4)

        self.apply(weights_init_)

        # action rescaling
        low = 0
        high = 1
        device = self.config.device
        self.action_scale = torch.FloatTensor(torch.ones(self.config.ACTION_SIZE) * (high - low) / 2.).to(device)
        self.action_bias = torch.FloatTensor(torch.ones(self.config.ACTION_SIZE) * (high + low) / 2.).to(device)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Enforcing Action Bound
        # the entropy is to be minimised
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        entropy = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, entropy, mean

    def backprop(self, loss):
        # backprop and gradient update
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimiser.step()


class Critic(nn.Module):

    def __init__(self, config):
        super(Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(config.STATE_SIZE + config.ACTION_SIZE, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(config.STATE_SIZE + config.ACTION_SIZE, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.apply(weights_init_)

        self.optimiser = optim.Adam(params=self.parameters(), lr=3e-4, weight_decay=1e-4)

    def forward(self, state, action):
        x = torch.concat([state, action], dim=1)
        q1 = self.relu(self.fc1(x))
        q1 = self.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = self.relu(self.fc4(x))
        q2 = self.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def q1(self, state, action):
        x = torch.concat([state, action], dim=1)
        q1 = self.leaky_relu(self.fc1(x))
        q1 = self.leaky_relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1

    def backprop(self, loss):
        # backprop and gradient update
        self.optimiser.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimiser.step()
