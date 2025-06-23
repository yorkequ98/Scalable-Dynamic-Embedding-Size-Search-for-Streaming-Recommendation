import numpy as np
import os
from Action import Action
import torch
from Config import Config
from Buffer import Buffer
from Evaluator import Evaluator
from dataset.DatasetLoader import DatasetLoader
from ActorCritic import Actor, Critic
from State import State
from IOManager import IOManager
from base_recsys.LightGCN import LightGCN


class Engine:

    def __init__(self, ID):
        config = Config(ID)

        self.config = config
        print(config.settings)

        self.io = IOManager(config)
        self.io.create_folders()
        self.io.delete_folder_contents('tmp/idx_embs/{}/'.format(self.config.ID))
        self.io.delete_folder_contents('tmp/model/{}/'.format(self.config.ID))

        self.loader = DatasetLoader(config, config.DATASET_NAME)

        self.evaluator = Evaluator(config)

        self.df_num = -1

        self.elbow_pint_ratio = 0.9

        self.state = State(config)

        self.actor = Actor(config).to(config.device)
        self.critic = Critic(config).to(config.device)

        self.target_critic = Critic(config).to(config.device)

        self.buffer = Buffer(config, self.actor, self.critic, self.target_critic)

        self.rewards = []
        self.ratios = []
        self.user_alphas = []
        self.item_alphas = []

        self.results = []
        # used to rescale the reward scores to (low, high)
        if config.DATASET_NAME == 'yelp':
            self.low = 0.7
            self.high = 1.2
        else:
            self.low = 0.9
            self.high = 1.1
        self.fine_tune_steps = 200
        self.decay_rate = 0.95

    def has_converged(self, num=5, threshold=1e-3):
        """
        Check if the policy has converged
        """
        series = self.ratios
        if len(series) < num:
            return False  # Convergence requires at least three terms
        last_three = series[-num:]  # Get the last N numbers from the series
        # Check if the absolute difference between consecutive terms is less than the threshold
        for i in range(1, len(last_three)):
            if abs(last_three[i] - last_three[i - 1]) > threshold:
                return False
        return True

    def report_final_results(self):
        index = 1
        for recall, ndcg in self.results:
            print("t{}: Recall@20 = {}, NGCD@20 = {}".format(index, recall, ndcg))
            index += 1
        avg_recall = np.mean([tup[0] for tup in self.results])
        avg_ndcg = np.mean([tup[1] for tup in self.results])
        print('AVG Recall@20 = {:.4f}; AVG NDCG@20 = {:.4f}'.format(avg_recall, avg_ndcg))

    def get_fresh_recommender(self, load_pretrain, size=None, user_sizes=None, item_sizes=None):
        """
        Initialise a new recommender
        Args:
            load_pretrain: whether to load pretrained embs from previous time segments
            size: assigning uniform size for each user and item
            user_sizes: sizes for each user
            item_sizes: sizes for each item

        Returns: a newly initialised recommender

        """
        if size is not None and user_sizes is not None:
            raise ValueError('either size or user_sizes has to be None!')
        config = self.config
        dataset = self.loader.datasets(self.df_num)
        if user_sizes is None:
            assert size is not None
            size = int(size)
            # fixed and uniform sizes
            user_sizes = np.ones(dataset.n_users, dtype=int) * size
            item_sizes = np.ones(dataset.n_items, dtype=int) * size
        recsys = LightGCN(config, dataset, user_sizes, item_sizes, load_pretrain).to(config.device)
        return recsys

    def init_dataset(self, t):
        if t > 0:
            self.loader.delete_dataset(t - 1)
        self.df_num = t
        seen_users = self.loader.seen_users(self.df_num)
        seen_items = self.loader.seen_items(self.df_num)
        self.n_users = len(seen_users)
        self.n_items = len(seen_items)
        self.state.ratio = self.n_users / (self.n_users + self.n_items)
        self.state.update_freqs(self.loader.datasets(self.df_num))
        print('Dataframe used: {}'.format(t), flush=True)
        print('#users seen: {}, #items seen: {}'.format(len(seen_users), len(seen_items)), flush=True)

    def train_one_batch(self, recsys):
        config = self.config
        dataset = self.loader.datasets(self.df_num)
        assert self.n_users == dataset.n_users
        users, pos_items, neg_items = dataset.train_loader(config.BATCH_SIZE)
        recsys.optimiser.zero_grad()
        total_loss = recsys.create_loss(
            users, pos_items, neg_items
        )
        total_loss.backward()
        recsys.optimiser.step()
        return total_loss

    def train_n_steps(self, recsys, step_num):
        total_loss = []
        for batch in range(step_num):
            batch_loss = self.train_one_batch(recsys)
            total_loss.append(batch_loss.item())
        return sum(total_loss) / step_num

    def train_till_convergence(self, recsys, dataset, path):
        print('Training recsys til convergence...')
        patience = self.config.MAX_PATIENCE
        best_metric, best_recall, best_ndcg = 0, 0, 0
        recsys.train()

        config = self.config
        last_loss = -1

        metrics = []
        for i in range(1, 1000000):
            loss = self.train_n_steps(recsys, self.fine_tune_steps)
            metric, msg, mean_recall, mean_ndcg = self.evaluator.eval_rec(recsys, dataset)
            metrics.append(metric)

            lr = recsys.optimiser.param_groups[0]['lr']
            recsys.optimiser.param_groups[0]['lr'] = max(config.MIN_LR, self.decay_rate * lr)
            print('i: {}: LR: {:.4f} loss: {:.4f}'.format(
                i * self.fine_tune_steps, recsys.optimiser.param_groups[0]['lr'], loss
            ))
            if metric >= best_metric:
                print(msg)
                best_metric = metric
                best_recall = mean_recall
                best_ndcg = mean_ndcg
                patience = self.config.MAX_PATIENCE
                torch.save(recsys.state_dict(), path + 'model.pth')
            else:
                patience -= 1
                print('Current patience： {}'.format(patience))
                if patience <= 0 and (last_loss > 0 and loss / last_loss > 0.95):
                    break
            last_loss = loss
        recsys.load_state_dict(torch.load(path + 'model.pth'))
        return recsys, best_recall, best_ndcg, metrics[0]

    def step(self, action, conv_fitness):
        dataset = self.loader.datasets(self.df_num)

        user_sizes, item_sizes = action.get_emb_sizes(dataset)

        recsys = self.get_fresh_recommender(load_pretrain=False, user_sizes=user_sizes, item_sizes=item_sizes)

        self.train_n_steps(recsys, self.fine_tune_steps)

        curr_fitness, msg, _, _ = self.evaluator.eval_rec(recsys, dataset)

        reward = curr_fitness / conv_fitness
        rescaled_reward = 10 * ((reward - self.low) / (self.high - self.low))
        print('Reward: {} / {} = {} ==> {}'.format(curr_fitness, conv_fitness, reward, rescaled_reward))

        # update state
        self.state.update_state(action, dataset, reward, recsys)

        # get new state
        next_state = self.state.get_current_state(self.n_users, self.n_items)
        return rescaled_reward, next_state

    def policy(self, prev_state, correct, mode):
        config = self.config
        if mode == 'exploration':
            output, _, mean = self.actor.sample(prev_state)
            mean = mean.cpu().detach().numpy()
            ratio, user_alpha, item_alpha = mean[0]
            self.ratios.append(ratio)
            self.user_alphas.append(user_alpha)
            self.item_alphas.append(item_alpha)
        else:
            assert mode == 'exploitation'
            _, _, output = self.actor.sample(prev_state)

        action = Action(config, output, self.n_users, self.n_items, correct)
        # add noise to the action and normalise it
        return action

    def update_policy(self, max_fitness):
        step = 0
        while True:
            print('-' * 25 + 'step {}'.format(step) + '-' * 25, flush=True)

            prev_state = self.state.get_current_state(self.n_users, self.n_items)

            action = self.policy(prev_state, correct=False, mode='exploration')

            reward, next_state = self.step(action, max_fitness)

            self.rewards.append(reward)

            self.buffer.add(prev_state, action, reward, next_state)

            for _ in range(3):
                self.buffer.learn()
            step += 1
            if self.has_converged(num=5, threshold=0.01) and step > 10:
                break

        for li in [self.rewards, self.ratios, self.user_alphas, self.item_alphas]:
            li.clear()
        print('-'*25 + 'end' + '-'*25)

    def update_recsys(self):
        config = self.config
        dataset = self.loader.datasets(self.df_num)

        if self.df_num == 0:
            # in segment 0, uniform sizes are used
            if config.BUDGET_TYPE == 'c':
                size = self.config.BUDGET
            else:
                size = config.BUDGET / (self.n_users + self.n_items)
            recsys = self.get_fresh_recommender(load_pretrain=False, size=size)
        else:
            prev_state = self.state.get_current_state(self.n_users, self.n_items)
            action = self.policy(prev_state, correct=True, mode='exploitation')
            user_sizes, item_sizes = action.get_emb_sizes(dataset)
            recsys = self.get_fresh_recommender(load_pretrain=True, user_sizes=user_sizes, item_sizes=item_sizes)

            np.save('tmp/sizes/user_sizes_{}_{}_{}_{}_{}_{}.npy'.format(
                config.DATASET_NAME, config.BUDGET, self.df_num,
                self.config.BUDGET_TYPE, self.config.RESERVOIR_SIZE, self.config.DISTRIBUTION
            ), user_sizes)
            np.save('tmp/sizes/item_sizes_{}_{}_{}_{}_{}_{}.npy'.format(
                config.DATASET_NAME, config.BUDGET, self.df_num,
                self.config.BUDGET_TYPE, self.config.RESERVOIR_SIZE, self.config.DISTRIBUTION
            ), item_sizes)

        recsys, mean_recall, mean_ndcg, conv_fitness = self.train_till_convergence(
            recsys, dataset, 'tmp/model/{}/'.format(config.ID)
        )
        self.io.save_embs(recsys, dataset)
        self.state.update_mean_embs(recsys)
        if self.df_num > 0:
            self.results.append((mean_recall, mean_ndcg))
        return conv_fitness

    def streaming_size_search(self):

        conv_fitness = -1
        self.init_dataset(0)
        if self.config.DATASET_NAME == 'amazon-book':
            assert self.loader.num_of_splits() - 1 == 8
        else:
            assert self.loader.num_of_splits() - 1 == 10

        for t in range(self.loader.num_of_splits()):
            print('*' * 20 + ' t = {} '.format(t) + '*' * 20)
            # update the policy
            if t > 0:
                self.update_policy(conv_fitness)
                # increment dataset
                try:
                    self.init_dataset(t)
                except:
                    break
            conv_fitness = self.update_recsys()
        self.report_final_results()


