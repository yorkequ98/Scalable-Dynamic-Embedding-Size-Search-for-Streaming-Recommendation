import numpy as np
import math
import torch


class Evaluator:

    def __init__(self, config):
        self.config = config

    def recall_at_k(self, zipped, total, k):
        top_k = zipped[:k]
        hits = np.sum([tup[0] for tup in top_k])
        return hits / np.maximum(total, 1)

    def ndcg_at_k(self, zipped, ideal_rank, k):
        top_k = zipped[:k]
        ranked = np.array([tup[0] for tup in top_k])
        positions = np.log2(np.arange(len(top_k)) + 2)
        dcg = np.round(ranked) / positions
        idcg = ideal_rank[:k] / positions
        return np.sum(dcg) / np.maximum(np.sum(idcg), 1)

    def process_ranking_metrics(self, recalls_5, recalls_10, recalls_20, ndcgs_5, ndcgs_10, ndcgs_20):
        metrics_arr = np.array([recalls_5, recalls_10, recalls_20, ndcgs_5, ndcgs_10, ndcgs_20])
        metircs_at_20 = np.array([recalls_20, ndcgs_20])

        mean_recall_5, mean_recall_10, mean_recall_20, \
            mean_ndcg_5, mean_ndcg_10, mean_ndcg_20 = np.mean(metrics_arr, axis=1)

        mean_metrics_per_entity = np.mean(metircs_at_20, axis=0)
        avg = np.mean(mean_metrics_per_entity)

        msg = 'R@5 = {:.4f}, N@5 = {:.4f}; R@10 = {:.4f}, N@10 = {:.4f}; R@20 = {:.4f}, N@20 = {:.4f}'
        msg = msg.format(mean_recall_5, mean_ndcg_5, mean_recall_10, mean_ndcg_10, mean_recall_20, mean_ndcg_20)
        return avg, msg, mean_recall_20, mean_ndcg_20

    def eval_rec(self, recsys, dataset):
        recsys.eval()
        sampled_users = np.array(dataset.test_user_vocab)
        sampled_items = np.array(dataset.test_item_vocab)
        recalls_20, ndcgs_20, recalls_10, ndcgs_10, recalls_5, ndcgs_5 = [], [], [], [], [], []

        num_chunks = 1
        chunk_size = math.ceil(len(sampled_users) / num_chunks)
        for chunk in range(num_chunks):
            start_ind = chunk * chunk_size
            end_ind = min(len(sampled_users), (chunk + 1) * chunk_size)
            users_in_chunk = sampled_users[start_ind: end_ind]

            y_pred, topk_ind = self.get_y_pred(recsys, users_in_chunk, sampled_items, dataset)

            assert len(y_pred) == len(users_in_chunk)
            for user_id in users_in_chunk:
                # assert dataset.test_user_vocab[user_id] == user_id
                total = np.sum(dataset.get_y_true_by_user(user_id)[sampled_items])
                # the position of user_id
                user_pos = np.asarray(users_in_chunk == user_id).nonzero()
                assert len(user_pos[0]) == 1
                user_pos = user_pos[0][0]
                assert dataset.get_y_true_by_user(user_id)[sampled_items].shape == y_pred[user_pos].shape

                # dataset.get_y_true_by_user(user_id) is only one single row
                y_true_selected = dataset.get_y_true_by_user(user_id)[sampled_items]
                # y_pred_selected is only a single row
                y_pred_selected = y_pred[user_pos]
                # topk indices associated with this specific user
                selected_topk_ind = topk_ind[user_pos]
                zipped = list(zip(y_true_selected[selected_topk_ind], y_pred_selected[selected_topk_ind]))

                recalls_20.append(self.recall_at_k(zipped, total, k=20))
                recalls_10.append(self.recall_at_k(zipped, total, k=10))
                recalls_5.append(self.recall_at_k(zipped, total, k=5))

                ideal_rank = np.sort(dataset.get_y_true_by_user(user_id)[sampled_items])[::-1]

                ndcgs_20.append(self.ndcg_at_k(zipped, ideal_rank, k=20))
                ndcgs_10.append(self.ndcg_at_k(zipped, ideal_rank, k=10))
                ndcgs_5.append(self.ndcg_at_k(zipped, ideal_rank, k=5))
        avg, msg, mean_recall, mean_ndcg = self.process_ranking_metrics(recalls_5, recalls_10, recalls_20, ndcgs_5, ndcgs_10, ndcgs_20)
        return avg, msg, mean_recall, mean_ndcg

    def get_y_pred(self, recsys, sampled_users, sampled_items, dataset):
        """Score all items for test users.
        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """
        config = self.config
        batch_size = 2000

        with torch.no_grad():
            user_ids = sampled_users
            n_user_batchs = len(user_ids) // batch_size + 1
            test_scores = np.array([])

            for u_batch_id in range(n_user_batchs):
                start = u_batch_id * batch_size
                end = min((u_batch_id + 1) * batch_size, len(user_ids))
                user_batch = user_ids[start: end]
                batch_users_gpu = torch.Tensor(user_batch).long().to(config.device)
                batch_items_gpu = torch.Tensor(sampled_items).long().to(config.device)
                ratings = recsys.get_users_rating(batch_users_gpu, batch_items_gpu).squeeze()
                if len(test_scores) == 0:
                    test_scores = ratings.cpu().numpy()
                else:
                    test_scores = np.concatenate((test_scores, ratings.cpu().numpy()), axis=0)

            # shape check
            assert test_scores.shape[0] == len(sampled_users) and test_scores.shape[1] == len(sampled_items)

            sampled_R = dataset.R.tocsr()[sampled_users][:, sampled_items]
            test_scores += sampled_R * -np.inf
            test_scores = torch.Tensor(test_scores)
            _, topk_ind = torch.topk(test_scores, k=20)

            topk_shape = topk_ind.size()
            assert topk_shape[0] == len(sampled_users) and topk_shape[1] == 20
            return test_scores.numpy(), topk_ind.numpy()

