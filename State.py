import numpy as np
import torch
import math

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


class State:

    def __init__(self, config):
        self.config = config

        self.ratio = 0.5

        if config.DISTRIBUTION in ['powerlaw', 'pareto']:
            self.user_alpha = 0.99
            self.item_alpha = 0.99
        elif config.DISTRIBUTION in ['normal']:
            self.user_alpha, self.item_alpha = 0.01, 0.01

        self.user_pvals = np.ones(config.USER_GROUP_NUM)
        self.item_pvals = np.ones(config.ITEM_GROUP_NUM)

        self.user_pvals = self.user_pvals / (np.sum(self.user_pvals) + np.sum(self.item_pvals))
        self.item_pvals = self.item_pvals / (np.sum(self.user_pvals) + np.sum(self.item_pvals))
        self.mean_emb = np.random.normal(0, 0.01, size=(config.MAX_EMB_SIZE * 2,))
        self.reward = 0.5
        self.user_item_ratio = 0.5

    def minmax_norm(self, vals):
        max_val = np.max(vals)
        min_val = np.min(vals)
        if max_val == min_val:
            return np.ones_like(vals)
        return (vals - min_val) / (max_val - min_val)

    def mean_pooling(self, data, group_num):
        group_length = math.ceil(len(data) / group_num)
        pooled = []
        for i in range(group_num):
            window = data[i * group_length: (i + 1) * group_length]
            if sum(window) == 0:
                pooled.append(0)
            else:
                pooled.append(np.mean(window))
        return np.array(pooled)

    def get_current_state(self, n_user, n_item):
        user_freqs = self.minmax_norm(self.user_freqs)
        item_freqs = self.minmax_norm(self.item_freqs)

        user_freq_gini = gini(user_freqs)
        item_freq_gini = gini(item_freqs)

        user_freqs = self.mean_pooling(user_freqs, self.config.USER_GROUP_NUM)
        item_freqs = self.mean_pooling(item_freqs, self.config.ITEM_GROUP_NUM)

        if self.config.BUDGET_TYPE == 'c':
            total_param = (n_user + n_item) * self.config.BUDGET
        else:
            total_param = self.config.BUDGET
        diff = (self.config.MAX_EMB_SIZE - self.config.MIN_EMB_SIZE)
        user_size_gini = self.ratio * (np.max(self.user_pvals) - np.min(self.user_pvals)) * total_param / diff
        item_size_gini = (1 - self.ratio) * (np.max(self.item_pvals) - np.min(self.item_pvals)) * total_param / diff

        self.user_item_ratio = n_user / (n_user + n_item)

        if self.config.STATE == 0:
            tmp = [
                user_freqs, item_freqs,
                # user_freq_gini, item_freq_gini,
                self.reward, self.user_item_ratio,
                self.mean_emb,
                user_size_gini, item_size_gini
            ]
        elif self.config.STATE == 1:
            tmp = [
                user_freqs, item_freqs,
                user_freq_gini, item_freq_gini,
                self.reward, self.user_item_ratio,
                self.mean_emb,
                # user_size_gini, item_size_gini
            ]
        elif self.config.STATE == 2:
            tmp = [
                user_freqs, item_freqs,
                user_freq_gini, item_freq_gini,
                self.reward, self.user_item_ratio,
                # self.mean_emb,
                user_size_gini, item_size_gini
            ]
        else:
            tmp = [
                user_freqs, item_freqs,
                user_freq_gini, item_freq_gini,
                self.reward, self.user_item_ratio,
                self.mean_emb,
                user_size_gini, item_size_gini
            ]
        state = []
        for i in range(len(tmp)):
            state.append(np.array(tmp[i]).reshape((1, -1)))
        state = np.concatenate(state, axis=1)

        return torch.tensor(state).float().to(self.config.device)

    def update_freqs(self, dataset):
        user_freqs = []
        item_freqs = []
        for user in dataset.user_vocab:
            user_freqs.append(dataset.user_freqs[user])

        for item in dataset.item_vocab:
            item_freqs.append(dataset.item_freqs[item])

        user_freqs = sorted(user_freqs, reverse=True)
        item_freqs = sorted(item_freqs, reverse=True)
        assert user_freqs[0] > user_freqs[-1]

        self.user_freqs = np.array(user_freqs)
        self.item_freqs = np.array(item_freqs)

    def update_mean_embs(self, recsys):
        user_embs, item_embs = recsys.get_mean_embedding()
        self.mean_emb = np.concatenate([user_embs, item_embs], axis=0)

    def update_state(self, action, dataset, reward, recsys):
        user_embs, item_embs = recsys.get_mean_embedding()
        self.mean_emb = np.concatenate([user_embs, item_embs], axis=0)

        self.ratio = action.ratio

        self.user_alpha = action.user_alpha
        self.item_alpha = action.item_alpha

        self.user_pvals = action.user_pvals
        self.item_pvals = action.item_pvals

        self.update_freqs(dataset)

        self.reward = np.array([reward])

        self.user_item_ratio = np.array([dataset.n_users / (dataset.n_users + dataset.n_items)])

