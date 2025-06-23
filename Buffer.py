import torch
import numpy as np
import torch.nn as nn


class Buffer:

    def __init__(self, config, actor, critic, target_critic):
        self.state_buffer = np.zeros((config.BUFFER_CAPACITY, config.STATE_SIZE))
        self.action_buffer = np.zeros((config.BUFFER_CAPACITY, config.ACTION_SIZE))
        self.reward_buffer = np.zeros((config.BUFFER_CAPACITY, 1))
        self.next_state_buffer = np.zeros((config.BUFFER_CAPACITY, config.STATE_SIZE))
        self.buffer_counter = 0
        self.alpha = config.ENTROPY_ALPHA

        self.config = config
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic

        self.rl_update_count = 0

        self.ps = []

        self.criterion = nn.MSELoss()

    def sample(self):
        device = self.config.device
        # Randomly sample indices
        record_range = min(self.buffer_counter, self.config.BUFFER_CAPACITY)

        if self.buffer_counter <= 1:
            ps = np.array([1])
        else:
            ps = np.arange(self.buffer_counter) + 1
            ps = ps / np.sum(ps)

        num_samples = 1
        if self.buffer_counter > 12:
            num_samples = 8

        batch_indices = np.random.choice(record_range, num_samples, p=ps, replace=False)
        # Convert to tensors
        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32).to(device)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32).to(device)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32).to(device)
        return state_batch, action_batch, reward_batch, next_state_batch

    def add(self, prev_state, action, reward, next_state):
        prev_state = prev_state.cpu().detach().numpy()
        action = action.array()
        next_state = next_state.cpu().detach().numpy()
        index = self.buffer_counter % self.config.BUFFER_CAPACITY
        self.state_buffer[index: index + 1] = prev_state
        self.action_buffer[index: index + 1] = action
        self.reward_buffer[index: index + 1] = reward
        self.next_state_buffer[index: index + 1] = next_state
        self.buffer_counter += 1

    def learn(self):
        def update_target(models):
            for target_model, model in models:
                target_weights = target_model.parameters()
                weights = model.parameters()
                for (a, b) in zip(target_weights, weights):
                    a.data.copy_(b.data * config.TAU + a.data * (1 - config.TAU))

        config = self.config
        self.rl_update_count += 1
        state_batch, action_batch, reward_batch, next_state_batch = self.sample()

        with torch.no_grad():
            next_state_action, nex_state_log_pi, _ = self.actor.sample(next_state_batch)
            # compute target critic values
            Q1, Q2 = self.target_critic.forward(next_state_batch, next_state_action)
            target_critic_val = torch.min(Q1, Q2) - self.alpha * nex_state_log_pi
            y = reward_batch + config.GAMMA * target_critic_val
        critic_value_1, critic_value_2 = self.critic.forward(state_batch, action_batch)
        critic_loss = self.criterion(critic_value_1, y) + self.criterion(critic_value_2, y)
        self.critic.backprop(critic_loss)
        message = 'Critic value: {:.4f}, reward = {:.4f}, critic loss = {:.4f}'
        print(message.format(critic_value_1.mean(), reward_batch.mean(), critic_loss.mean()), flush=True)

        if self.rl_update_count % 1 == 0:
            # update the actor
            pi, log_pi, _ = self.actor.sample(state_batch)

            qf1_pi, qf2_pi = self.critic.forward(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            # this term will be minimised, namely the critic value should be maximised
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            message = 'Actor loss: {} - {} = {}'.format(
                (self.alpha * log_pi).mean().item(), min_qf_pi.mean().item(), actor_loss.item()
            )
            print(message, flush=True)

            self.actor.backprop(actor_loss)
        # soft updates
        update_target([(self.target_critic, self.critic)])

    def clear(self):
        config = self.config
        self.state_buffer = np.zeros((config.BUFFER_CAPACITY, config.STATE_SIZE))
        self.action_buffer = np.zeros((config.BUFFER_CAPACITY, config.ACTION_SIZE))
        self.reward_buffer = np.zeros((config.BUFFER_CAPACITY, 1))
        self.next_state_buffer = np.zeros((config.BUFFER_CAPACITY, config.STATE_SIZE))
        self.buffer_counter = 0

    def __len__(self):
        return self.buffer_counter
