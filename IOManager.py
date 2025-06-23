import numpy as np
import torch
import os
import shutil


class IOManager:

    def __init__(self, config):
        self.config = config

    def create_folders(self):
        config = self.config
        folders_to_create = [
            'tmp/',
            'tmp/idx_embs/',
            'tmp/sizes/'
            'tmp/datasets/{}/'.format(config.DATASET_NAME),
            'tmp/idx_embs/{}'.format(config.ID),
            'tmp/datasets/{}/{}/'.format(config.DATASET_NAME, config.NUM_SPLITS),
            'tmp/datasets/{}/{}/adj_matrices/'.format(config.DATASET_NAME, config.NUM_SPLITS),
            'tmp/model/{}/'.format(config.ID)
        ]
        for folder in folders_to_create:
            try:
                os.makedirs(folder)
            except Exception:
                pass

    def save_embs(self, recsys, dataset):
        ID = self.config.ID
        # save embs
        user_embs = recsys.embedding_user.weight * recsys.user_mask
        torch.save(user_embs, 'tmp/idx_embs/{}/user_embs.pt'.format(ID))
        item_embs = recsys.embedding_item.weight * recsys.item_mask
        torch.save(item_embs, 'tmp/idx_embs/{}/item_embs.pt'.format(ID))

        # save global indices and local indices
        with open('tmp/idx_embs/{}/user_idx.csv'.format(ID), 'wb') as f:
            np.save(f, np.array(dataset.user_vocab))

        with open('tmp/idx_embs/{}/item_idx.csv'.format(ID), 'wb') as f:
            np.save(f, np.array(dataset.item_vocab))

    def delete_folder_contents(self, folder_path):
        try:
            # Check if the folder exists
            if os.path.exists(folder_path):
                # Iterate over all the files and subdirectories in the folder
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)

                    # Check if the item is a file, then remove it
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    # If the item is a directory, remove it recursively
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
        except Exception as e:
            print(f"An error occurred: {e}")


