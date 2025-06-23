import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LightGCN(nn.Module):
    def __init__(self, config, dataset, user_sizes, item_sizes, load_pretrain):
        super(LightGCN, self).__init__()
        self.config = config
        self.droput = False

        self.num_users = dataset.n_users
        self.num_items = dataset.n_items

        self.update_sizes(user_sizes, item_sizes)

        self.n_layers = 3
        self.keep_prob = 0.6
        self.A_split = False
        self.decay = 1e-4
        self.init_weight(load_pretrain)

        self.Graph = dataset.create_norm_adj_mat()
        self.Graph = dataset.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(config.device)

        self.optimiser = optim.Adam(self.parameters(), lr=config.MAX_LR)

    def __iterate_and_update(self, sizes):
        mask = np.zeros((len(sizes), self.config.MAX_EMB_SIZE), dtype=np.int32)
        for r in range(len(sizes)):
            mask[r][:sizes[r]] = 1
        return torch.tensor(mask, device=self.config.device)

    def update_sizes(self, user_sizes, item_sizes):
        self.user_sizes = user_sizes
        self.item_sizes = item_sizes
        self.user_mask = self.__iterate_and_update(self.user_sizes)
        self.item_mask = self.__iterate_and_update(self.item_sizes)

    def init_weight(self, load_pretrained):
        ID = self.config.ID
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.config.MAX_EMB_SIZE
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.config.MAX_EMB_SIZE
        )
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)

        if load_pretrained:
            try:
                # index mapping of previous dfs
                with open('tmp/idx_embs/{}/user_idx.csv'.format(ID), 'rb') as f:
                    user_idx = np.load(f)
                with open('tmp/idx_embs/{}/item_idx.csv'.format(ID), 'rb') as f:
                    item_idx = np.load(f)
                user_idx = torch.tensor(user_idx).long()
                item_idx = torch.tensor(item_idx).long()

                user_path = 'tmp/idx_embs/{}/user_embs.pt'.format(ID)
                item_path = 'tmp/idx_embs/{}/item_embs.pt'.format(ID)
                user_embs = torch.load(user_path).to('cpu')
                item_embs = torch.load(item_path).to('cpu')

                with torch.no_grad():
                    print('Pretrained embedding loaded')
                    self.embedding_user.weight.data[user_idx] = user_embs
                    self.embedding_item.weight.data[item_idx] = item_embs
            except Exception as e:
                print('Loading presaved embedding failed!')
                print(e)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def get_users_rating(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def get_mean_embedding(self):
        users_emb = self.embedding_user.weight * self.user_mask
        items_emb = self.embedding_item.weight * self.item_mask
        return torch.mean(users_emb, 0).cpu().detach().numpy(), torch.mean(items_emb, 0).cpu().detach().numpy()

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight * self.user_mask
        items_emb = self.embedding_item.weight * self.item_mask

        all_emb = torch.cat([users_emb, items_emb]).float()
        embs = [all_emb]
        if self.droput:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users) * self.user_mask[users]
        pos_emb_ego = self.embedding_item(pos_items) * self.item_mask[pos_items]
        neg_emb_ego = self.embedding_item(neg_items) * self.item_mask[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def create_loss(self, users, pos_items, neg_items):
        users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego = self(users, pos_items, neg_items)

        reg_loss = (users_emb_ego.norm(2).pow(2) +
                    pos_emb_ego.norm(2).pow(2) +
                    neg_emb_ego.norm(2).pow(2)) / float(len(users)) * self.decay

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        mf_loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        total_loss = mf_loss + reg_loss
        return total_loss

