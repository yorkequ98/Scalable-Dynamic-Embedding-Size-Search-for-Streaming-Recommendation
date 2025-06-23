import torch
import os
import sys
import numpy as np
import random
from scipy.stats import powerlaw, truncnorm, pareto

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


# Specify the settings here
# {budget_type: c, budget: 32} means the mean embedding size is fixed to be 32. (fixed mean size)
# {budget_type: B, budget: 10} means the memory size of the embeddings is fixed to be 10MB. (fixed memory budget)
# {state: 0} specifies what state incorporates. See State.py
ID_details = {
    0: {
        'dataset_name': 'amazon-book', 'budget': 32, 'state': 0,
        'distribution': 'powerlaw', 'group_num': 256, 'reservoir_size': 0.2, 'budget_type': 'B'
    },
    1: {
        'dataset_name': 'yelp', 'budget': 10, 'state': 0,
        'distribution': 'powerlaw', 'group_num': 256, 'reservoir_size': 0.2, 'budget_type': 'c'
    }
}


class Config:

    def __init__(self, ID):
        self.ID = ID
        settings = ID_details[ID]
        self.settings = settings
        self.DATASET_NAME = settings['dataset_name']
        self.BUDGET = settings['budget']
        self.NUM_SPLITS = 'variable'

        self.RESERVOIR_SIZE = settings['reservoir_size']
        self.MAX_EMB_SIZE = 256
        self.MIN_EMB_SIZE = 1
        self.NEG_SAMPLES = 1
        self.BUDGET_TYPE = settings['budget_type']
        if self.BUDGET_TYPE == 'B':
            self.BUDGET = int(self.BUDGET * 8e6 / 32)

        self.STATE = settings['state']
        self.USER_GROUP_NUM = settings['group_num']
        self.ITEM_GROUP_NUM = self.USER_GROUP_NUM
        self.STATE = settings['state']
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.BUFFER_CAPACITY = 10000
        self.ACTION_SIZE = 3
        self.STATE_SIZE = 6 + 2 * self.MAX_EMB_SIZE + self.USER_GROUP_NUM + self.ITEM_GROUP_NUM
        if self.STATE in [0, 1]:
            self.STATE_SIZE = 4 + 2 * self.MAX_EMB_SIZE + self.USER_GROUP_NUM + self.ITEM_GROUP_NUM
        elif self.STATE == 2:
            self.STATE_SIZE = 6 + self.USER_GROUP_NUM + self.ITEM_GROUP_NUM
        self.DISTRIBUTION = settings['distribution']
        self.MAX_ALPHA = {'powerlaw': 30, 'pareto': 100, 'normal': 1e-5}[self.DISTRIBUTION]
        self.MIN_ALPHA = {'powerlaw': 0.1, 'pareto': 30, 'normal': 1e-6}[self.DISTRIBUTION]
        self.ENTROPY_ALPHA = 0.2
        self.USER_ITEM_SEPARATION = True
        if not self.USER_ITEM_SEPARATION:
            self.STATE_SIZE = 450
            self.ACTION_SIZE = 1

        if len(sys.argv) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
        GPU = torch.cuda.is_available()
        self.device = torch.device('cuda' if GPU else "cpu")
        self.SEED = seed
        self.BATCH_SIZE = 10000
        self.MAX_PATIENCE = 3
        self.DECAY_BATCHES = 200
        self.MAX_LR = 0.03
        self.MIN_LR = 0.001
        self.K_CORE = {'amazon-book': 30, 'ml-25m': 30, 'ml-10m': 5, 'yelp': 10}[self.DATASET_NAME]

    def powerlaw(self, alpha, num):
        if alpha < self.MIN_ALPHA or alpha > self.MAX_ALPHA:
            raise ValueError('Invalid value of alpha: ', alpha)
        pvals = []
        for _ in range(100):
            r = powerlaw.rvs(a=alpha, size=num)
            r = np.sort(r)[::-1]
            r = r / np.sum(r)
            pvals.append(r)
        return np.mean(pvals, axis=0)

    def pareto(self, alpha, num):
        if alpha < self.MIN_ALPHA or alpha > self.MAX_ALPHA:
            raise ValueError('Invalid value of alpha: ', alpha)
        pvals = []
        for _ in range(100):
            r = pareto.rvs(size=num, b=alpha)
            r = np.sort(r)[::-1]
            r = r / np.sum(r)
            pvals.append(r)
        return np.mean(pvals, axis=0)

    def normal(self, alpha, num):
        if alpha < self.MIN_ALPHA or alpha > self.MAX_ALPHA:
            raise ValueError('Invalid value of alpha: ', alpha)
        pvals = []
        mean = 1/num
        a = -mean / alpha
        b = (1 - mean) / alpha
        for _ in range(100):
            r = truncnorm.rvs(a=a, b=b, loc=1/num, scale=alpha, size=num)
            r = np.sort(r)[::-1]
            r = r / np.sum(r)
            pvals.append(r)
        return np.mean(pvals, axis=0)

