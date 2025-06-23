import pandas as pd
import numpy as np
import random


class Reservoir:

    def __init__(self, config, capacity, users):
        self.config = config
        self.content = pd.DataFrame()
        self.probs = []
        self.capacity = capacity
        self.users = set(users)

    def __len__(self):
        return len(self.content)

    def add_content(self, df):
        df = df[df['userID'].isin(self.users)]
        if len(df) > self.capacity:
            df = df.sample(n=min(int(self.capacity * 1.1), len(df)), replace=False).sort_index()
        assert len(self.content) == 0

        # add entries to the reservoir if it is not full
        if len(df) <= self.capacity:
            self.content = pd.concat([self.content, df], axis=0)
        else:
            # keep adding entries to the reservoir till it is full
            self.content = pd.concat([self.content, df.iloc[:self.capacity]], axis=0)
            assert len(self.content) == self.capacity

            to_be_replaced = []
            entries = pd.DataFrame()
            for t in range(self.capacity, len(df)):

                rand = random.randint(0, t + 1)
                if rand < self.capacity:
                    entry = df.iloc[[t]]
                    to_be_replaced.append(rand)
                    entries = pd.concat([entries, entry], axis=0)

            self.content.iloc[np.array(to_be_replaced)] = entries
