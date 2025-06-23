import pandas as pd
import numpy as np
from dataset.split_utils import filter_k_core
import json
import pickle as pkl
from Reservoir import Reservoir
from dataset.ImplicitCF import ImplicitCF
from datetime import datetime
import os


class DatasetLoader:

    def __init__(self, config, dataset_name):
        self.config = config
        # create datasets
        self._datasets = []
        try:
            directory = os.fsencode('tmp/datasets/{}/{}/'.format(config.DATASET_NAME, config.NUM_SPLITS))
            num_segs = 0
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".obj"):
                    _, _, _, base_model, reservoir_size = filename.split('_')
                    reservoir_size = float(reservoir_size.split('.obj')[0])
                    if reservoir_size == config.RESERVOIR_SIZE:
                        num_segs += 1
            if num_segs == 0:
                raise FileNotFoundError('No dataset presaved. Now computing data segments...')

            for t in range(num_segs):
                file = open('tmp/datasets/{}/{}/ds_at_{}_LightGCN_{}.obj'.format(
                    config.DATASET_NAME, config.NUM_SPLITS, t, config.RESERVOIR_SIZE), 'rb'
                )
                ds_at_t = pkl.load(file)
                file.close()
                print('#users: {}, #items: {} in the training set in time segment {}'.format(
                    ds_at_t.n_users, ds_at_t.n_items, t), flush=True
                )
                self._datasets.append(ds_at_t)

        except Exception as e:
            print(e)
            dfs = self.compute_dfs(dataset_name)
            for t in range(len(dfs)):
                prev_df = None
                if t > 0:
                    prev_df = pd.concat(dfs[:t], axis=0)
                    prev_df = prev_df.drop_duplicates()
                ds_at_t = ImplicitCF(config, prev_df, dfs[t], t)
                file = open('tmp/datasets/{}/{}/ds_at_{}_LightGCN_{}.obj'.format(
                    config.DATASET_NAME, config.NUM_SPLITS, t, config.RESERVOIR_SIZE), 'wb'
                )
                pkl.dump(ds_at_t, file)
                file.close()
                print('#users: {}, #items: {} in the training set in time segment {}'.format(
                    ds_at_t.n_users, ds_at_t.n_items, t), flush=True
                )
                self._datasets.append(ds_at_t)

    def datasets(self, t):
        return self._datasets[t]

    def num_of_splits(self):
        return len(self._datasets)

    def delete_dataset(self, t):
        self._datasets[t] = None

    def seen_users(self, t):
        return self._datasets[t].user_vocab

    def seen_items(self, t):
        return self._datasets[t].item_vocab

    def get_n_entities(self, t):
        return len(self.seen_users(t)), len(self.seen_items(t))

    def compute_dfs(self, dataset_name):
        try:
            cols = ['userID', 'itemID', 'rating', 'timestamp']
            df = pd.read_csv(
                'data/{}/kcore.csv'.format(dataset_name), names=cols, sep=',', header=0
            )
        except Exception as e:
            print(e, flush=True)
            print('Loading dataset failed. Computing kcore dataset...', flush=True)
            df = self.create_df(dataset_name, kcore=True)
        # sort df
        df.sort_values(by=['timestamp'], inplace=True, ignore_index=True)
        # reindex
        df = self.data_processing(df)

        # split the df
        if self.config.NUM_SPLITS == 'variable':
            users = set()
            items = set()
            new_users = set()
            new_items = set()
            split_pos = []
            count = 0
            for index, row in df.iterrows():
                if row['userID'] not in users:
                    new_users.add(row['userID'])
                if row['itemID'] not in items:
                    new_items.add(row['itemID'])
                if len(new_users) + len(new_items) > 15000:
                    count += 1
                    split_pos.append(index)
                    users.update(new_users)
                    items.update(new_items)
                    assert len(users) - 1 == max(users) and len(items) - 1 == max(items)
                    new_users.clear()
                    new_items.clear()

            dfs = []
            for i in range(len(split_pos)):
                pos = split_pos[i]
                if i == 0:
                    prev_pos = 0
                else:
                    prev_pos = split_pos[i - 1]
                dfs.append(df.iloc[prev_pos: pos])
            dfs.append(df.iloc[split_pos[-1]:])
        else:
            dfs = np.array_split(df, self.config.NUM_SPLITS)

        # # add replay data from D_1 onward
        for t in range(1, len(dfs)):
            df = dfs[t]
            capacity = int(len(df) * self.config.RESERVOIR_SIZE)
            reservoir = Reservoir(self.config, capacity, df['userID'].unique())
            reservoir.add_content(pd.concat(dfs[:t], axis=0))
            dfs[t] = pd.concat([reservoir.content, df], axis=0)
        return dfs

    def create_df(self, dataset, kcore=False):
        if dataset in ['ml-10m', 'ml-1m']:
            filename = 'data/{}/ratings.dat'.format(dataset)
            header = ['userID', 'itemID', 'rating', 'timestamp']
            dtypes = {h: np.int32 for h in header}
            df = pd.read_csv(
                filename, sep='::', names=header, engine='python', dtype=dtypes, header=0
            )
        elif dataset == 'ml-25m':
            filename = 'data/ml-25m/ratings.csv'
            header = ['userID', 'itemID', 'rating', 'timestamp']
            dtypes = {h: np.int32 for h in header}
            df = pd.read_csv(
                filename, sep=',', header=0, names=header, engine='python', dtype=dtypes
            )
        elif dataset in ['amazon-book', 'amazon-cellphone']:
            filename = 'data/{}/ratings.json'.format(dataset)
            data = []
            cols = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
            with open(filename) as f:
                for line in f:
                    doc = json.loads(line)
                    lst = [doc['reviewerID'], doc['asin'], doc['overall'], doc['unixReviewTime']]
                    data.append(lst)
            df = pd.DataFrame(data=data, columns=cols)
            columns = {'reviewerID': 'userID', 'asin': 'itemID', 'overall': 'rating', 'unixReviewTime': 'timestamp'}
            df = df.rename(columns=columns)
        elif dataset == 'yelp':
            filename = 'data/{}/ratings.json'.format(dataset)
            data = []
            cols = ['reviewerID', 'asin', 'overall', 'date']
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    time = datetime.strptime(doc['date'], '%Y-%m-%d %H:%M:%S')
                    lst = [doc['user_id'], doc['business_id'], doc['stars'], time]
                    data.append(lst)
            df = pd.DataFrame(data=data, columns=cols)
            columns = {'reviewerID': 'userID', 'asin': 'itemID', 'overall': 'rating', 'date': 'timestamp'}
            df = df.rename(columns=columns)

        else:
            raise ValueError('{} is invalid!'.format(dataset))
        print('Before k-core processing, # users {}, # items {}'.format(len(df['userID'].unique()), len(df['itemID'].unique())))
        if kcore:
            k = self.config.K_CORE
            df = filter_k_core(self.config, df, core_num=k)
            assert min(df['userID'].value_counts()) >= k and min(df['itemID'].value_counts()) >= k
            df.to_csv('data/{}/kcore.csv'.format(dataset), encoding='utf-8', index=False)
        return df

    def data_processing(self, df):
        # replace IDs with indices
        user_idx = df[['userID']].drop_duplicates(keep='first').reindex()
        user_idx["userID_idx"] = np.arange(len(user_idx))

        item_idx = df[['itemID']].drop_duplicates(keep='first').reindex()
        item_idx["itemID_idx"] = np.arange(len(item_idx))

        df = pd.merge(df, user_idx, on="userID", how="left")
        df = pd.merge(df, item_idx, on="itemID", how="left")

        df_reindex = df[
            ["userID_idx", "itemID_idx", "rating", "timestamp"]
        ]
        df_reindex.columns = ["userID", "itemID", "rating", "timestamp"]
        return df_reindex
