import numpy as np
import math
import time
from scipy.stats import powerlaw, truncexpon, truncnorm, lognorm


class Action:

    def __init__(self, config, output, n_users, n_items, correct):
        self.config = config
        if config.BUDGET_TYPE == 'c':
            self.num_of_params = config.BUDGET * (n_users + n_items)
        else:
            self.num_of_params = config.BUDGET
        # all in the range of (0, 1)
        ratio, user_alpha, item_alpha = output.cpu().detach().numpy().reshape(-1)

        self.ratio = np.clip(ratio, a_min=1e-5, a_max=1-1e-5)
        self.user_alpha = np.clip(user_alpha, a_min=0, a_max=1)
        self.item_alpha = np.clip(item_alpha, a_min=0, a_max=1)

        # print('Ratio: {}. user alpha: {}, item alpha: {}'.format(ratio, user_alpha, item_alpha))
        del user_alpha, item_alpha

        # sizes of each user/item
        self.user_pvals, self.u_sizes = self.get_sizes(
            self.user_alpha.item(), self.ratio.item(), correct, n_users
        )
        self.item_pvals, self.i_sizes = self.get_sizes(
            self.item_alpha.item(), 1 - self.ratio.item(), correct, n_items
        )

    def get_sizes(self, alpha, ratio, correct, num):
        """
        Calculate the embedding sizes
        Args:
            alpha: parameter of the powerlaw distribution
            ratio: user or item ratio
            correct: whether to correct/normalise the action if it is in an illegal domain
            num: number of users or items

        Returns: flattened softmax scores and the number of parameters
        """
        config = self.config
        budget = ratio * self.num_of_params
        # rescale alpha
        alpha = alpha * (config.MAX_ALPHA - config.MIN_ALPHA) + config.MIN_ALPHA
        if config.DISTRIBUTION == 'exponential':
            pvals = config.exponential(alpha, num)
        elif config.DISTRIBUTION == 'powerlaw':
            pvals = config.powerlaw(alpha, num)
        elif config.DISTRIBUTION == 'normal':
            pvals = config.normal(alpha, num)
        elif config.DISTRIBUTION == 'pareto':
            pvals = config.pareto(alpha, num)
        else:
            raise ValueError('Invalid choice of distribution: ', config.DISTRIBUTION)
        pvals = pvals / np.sum(pvals)
        sizes = np.maximum(config.MIN_EMB_SIZE, pvals * budget)
        t0 = time.time()
        while correct and (np.max(sizes) > config.MAX_EMB_SIZE or np.min(sizes) < config.MIN_EMB_SIZE):
            if time.time() - t0 > 1:
                break
            sizes = np.clip(sizes, a_min=config.MIN_EMB_SIZE, a_max=config.MAX_EMB_SIZE)
            pvals = sizes / np.sum(sizes)
            sizes = pvals * budget
        sizes = np.round(sizes)
        return pvals, sizes

    def assign_sizes_by_importance(self, sizes, importance):
        # sort user/item IDs_importance pairs by their importance
        entity_importance = [(ID, freq) for ID, freq in importance]
        # Sort ID: importance pairs
        sorted_entities = sorted(entity_importance, key=lambda tup: (tup[1], tup[0]), reverse=True)
        # ID: size pairs
        assigned_sizes = []
        for i in range(len(sorted_entities)):
            ID = sorted_entities[i][0]
            size = sizes[i]
            # add sorted IDs and sorted sizes
            # the first entry is (ID, size), where ID is the most frequent entity and size is the largest size
            assigned_sizes.append((ID, size))
        # sort ID: size pairs to let ID=0 be the first element
        assigned_sizes = sorted(assigned_sizes, key=lambda tup: tup[0])
        assert len(assigned_sizes) == len(importance) == len(sorted_entities)
        assigned_sizes = np.array([assigned_sizes[entity][1] for entity in range(len(assigned_sizes))])
        return assigned_sizes

    def get_emb_sizes(self, dataset):
        # assign sizes to users and items according to their importance
        user_importance = []
        item_importance = []
        for user in dataset.user_freqs:
            user_importance.append((user, dataset.user_freqs[user]))
        for item in dataset.item_freqs:
            item_importance.append((item, dataset.item_freqs[item]))
        user_sizes = self.assign_sizes_by_importance(self.u_sizes, user_importance)
        item_sizes = self.assign_sizes_by_importance(self.i_sizes, item_importance)
        return user_sizes.astype(int), item_sizes.astype(int)

    def array(self):
        arr = np.concatenate([
            self.ratio.reshape((1, -1)),
            self.user_alpha.reshape((1, -1)),
            self.item_alpha.reshape((1, -1))
        ], 1)
        return arr

