import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from collections import Counter


class ImplicitCF(object):
    def __init__(self, config, prev_df, current_df, df_num):
        self.config = config
        self.df_num = df_num
        if prev_df is not None:
            # excluding the rows in current_df from prev_df
            prev_df = (
                pd.merge(prev_df, current_df, indicator=True, how='outer')
                .query('_merge=="left_only"')
                .drop('_merge', axis=1)
            )

        split_idx = round(len(current_df) * 0.8)
        # split df1 into train and test set
        if prev_df is not None:
            train = pd.concat([prev_df, current_df.iloc[:split_idx]], axis=0)
        else:
            train = current_df.iloc[:split_idx]
        test = current_df.iloc[split_idx:]

        # all the users and items in the training sets
        # should compute sizes for these users/items
        self.user_vocab = list(train['userID'].unique())
        self.item_vocab = list(train['itemID'].unique())
        self.n_users, self.n_items = len(self.user_vocab), len(self.item_vocab)

        # for testing purposes
        # only rank the following items for the following users
        self.test_user_vocab = list(set(test['userID'].values).intersection(train['userID'].values))
        self.test_item_vocab = list(set(test['itemID'].values).intersection(train['itemID'].values))

        # for sampling purposes
        # sampling from these users/items
        self.current_users = list(set(current_df.iloc[:split_idx]['userID'].unique()))
        self.current_items = list(set(current_df.iloc[:split_idx]['itemID'].unique()))

        # dynamic user/item frequencies
        # the number of appearances from the time they first appear to the end of the final timestamp
        self.user_freqs = Counter({user: 0 for user in self.user_vocab})
        self.user_freqs.update(Counter(current_df['userID'][:split_idx]))
        self.item_freqs = Counter({item: 0 for item in self.item_vocab})
        self.item_freqs.update(Counter(current_df['itemID'][:split_idx]))

        # setting up R and interacted_status
        self.interact_status, self.R = self._init_train_data(train, prev_df)

        # create y_true
        self.y_true = self.create_y_true(test)


    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def create_y_true(self, test):
        positive_pairs = []
        for user, item in test[['userID', 'itemID']].values:
            if user in self.test_user_vocab and item in self.test_item_vocab:
                positive_pairs.append((user, item))
        rows = []
        cols = []
        data = []
        for user_id, item_id in positive_pairs:
            rows.append(user_id)
            cols.append(item_id)
            data.append(1)
        y_true = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.int8)
        return y_true

    def get_y_true_by_user(self, user_id):
        return self.y_true[user_id].toarray()[0]

    def _init_train_data(self, train, prev_df):
        """
        Record items interated with each user in a dataframe self.interact_status, and create adjacency
        matrix self.R.
        """
        interact_status = (
            train.groupby('userID')['itemID']
            .apply(set)
            .reset_index()
            .rename(columns={'itemID': "itemID_interacted"})
        )
        if prev_df is not None:
            prev_interacted_items = (
                prev_df.groupby('userID')['itemID']
                .apply(set)
                .reset_index()
                .rename(columns={'itemID': 'prev_interacted_items'})
            )
            interact_status = interact_status.merge(prev_interacted_items, how='left')
        else:
            interact_status['prev_interacted_items'] = pd.Series([np.nan] * len(interact_status))

        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        R[train['userID'], train['itemID']] = 1.0
        return interact_status, R

    def create_norm_adj_mat(self):
        """Create normalized adjacency matrix.
        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.
        """
        config = self.config
        path = 'tmp/datasets/{}/{}/adj_matrices/norm_adj_mat_{}_{}_{}.npz'.format(
            config.DATASET_NAME, config.NUM_SPLITS, self.df_num, 'LightGCN', config.RESERVOIR_SIZE
        )
        try:
            norm_adj_mat = sp.load_npz(path)
        except Exception as e:
            print(e)
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R.tolil()
            adj_mat[: self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T

            adj_mat = adj_mat.todok()
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_mat.dot(d_mat_inv)

            sp.save_npz(path, norm_adj_mat)
        return norm_adj_mat.tocsr()

    def train_loader(self, batch_size):
        """Sample train data every batch. One positive item and one negative item sampled for each user.

        Args:
            batch_size (int): Batch size of users.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray:
            - Sampled users.
            - Sampled positive items.
            - Sampled negative items.
        """
        def sample_neg(x, y):
            if not isinstance(y, set):
                y = set()
            tried = []

            while True:
                neg_id = random.choice(self.current_items)
                tried.append(neg_id)
                if len(tried) == len(self.current_items):
                    neg_id = random.randint(0, self.n_items - 1)
                if neg_id not in x and neg_id not in y:
                    return neg_id

        def sample_pos(x, y):
            if not isinstance(y, set):
                y = set()
            diff = x.difference(y)
            list_x = list(diff)
            if len(list_x) == 0:
                list_x = list(x)
                return random.choice(list_x)
            while True:
                pos_id = random.choice(list_x)
                if pos_id not in y:
                    return pos_id

        config = self.config
        # only select users from the current df excluding the test set
        selected_indices = np.random.choice(
            self.current_users, batch_size, replace=len(self.current_users) < batch_size
        )
        interact = self.interact_status.iloc[selected_indices]

        # the items should come from the current df
        pos_items = interact.apply(
            lambda x: sample_pos(x["itemID_interacted"], x["prev_interacted_items"]), axis=1
        )

        # cannot sample the items from prev_df
        neg_items = interact.apply(
            lambda x: sample_neg(x["itemID_interacted"], x["prev_interacted_items"]), axis=1
        )

        users = np.array(interact['userID'])
        users = torch.tensor(users).long().to(config.device)

        pos_items = np.array(pos_items)
        pos_items = torch.tensor(pos_items).long().to(config.device)

        neg_items = np.array(neg_items)
        neg_items = torch.tensor(neg_items).long().to(config.device)

        return users, pos_items, neg_items
