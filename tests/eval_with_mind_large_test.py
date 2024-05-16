from poprox_recommender import handler, default
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
from os import path
import sys
import pandas as pd
from ast import literal_eval
from multiprocessing import Pool

print(os.getcwd())

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, args, news_path):
        super(NewsDataset, self).__init__()
        self.args = args
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + self.args.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(self.args.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item

class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, args, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.args = args
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0


    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:self.args.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = self.args.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item

class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item

def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4

@torch.no_grad()
def evaluate(args, model, directory, num_workers, max_count=sys.maxsize):
    """
    Make sure to run `dvc pull` first if you don't have the mind large test data in your development directory
    """
    news_dataset = NewsDataset(args, path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False,
                                 pin_memory=True)
    
    news2vector = {}
    for minibatch in tqdm(news_dataloader,
                          desc="Calculating vectors for news"):
        news_ids = minibatch["id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector
    
    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())

    user_dataset = UserDataset(args, path.join(directory, 'behaviors.tsv'),
                               path.join(directory,'user2int.tsv'))
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    for minibatch in tqdm(user_dataloader,
                          desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                              dim=0).transpose(0, 1)


            user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=args.num_workers)

    count = 0
    tasks = []

    for minibatch in tqdm(behaviors_dataloader,
                          desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break

        candidate_news_vector = torch.stack([
            news2vector[news[0].split('-')[0]]
            for news in minibatch['impressions']
        ],
                                            dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = model.get_prediction(candidate_news_vector,
                                                 user_vector)

        y_pred = click_probability.tolist()
        y_true = [
            int(news[0].split('-')[1]) for news in minibatch['impressions']
        ]

        tasks.append((y_true, y_pred))

    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(
        ndcg10s)

if __name__ == '__main__':
    # can modify in bash file
    import argparse

    parser = argparse.ArgumentParser(description='processing some parameters')
    parser.add_argument('--num_epochs', type=float, default=10)
    parser.add_argument('--num_batches_show_loss', type=float, default=100)
    parser.add_argument('--num_batches_validate', type=float, default=1000)
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=float, default=4)
    parser.add_argument('--num_clicked_news_a_user', type=float, default=50)
    parser.add_argument('--word_freq_threshold', type=float, default=1)
    parser.add_argument('--dropout_probability', type=float, default=0.2)
    parser.add_argument('--word_embedding_dim', type=float, default=300)
    parser.add_argument('--category_embedding_dim', type=float, default=100)
    parser.add_argument('--query_vector_dim', type=float, default=200)
    parser.add_argument('--num_attention_heads', type=float, default=15)
    parser.add_argument('--dataset_attributes', type=dict, default={
        "news": ['title'],
        "record": []
    })

    # modify during data_preprocess
    parser.add_argument('--negative_sampling_ratio', type=float, default=4)
    parser.add_argument('--num_words_title', type=float, default=20)
    parser.add_argument('--num_words_abstract', type=float, default=50)
    parser.add_argument('--num_users', type=float, default=1 + 50000)
    parser.add_argument('--num_words', type=float, default=1 + 70972)
    parser.add_argument('--num_categories', type=float, default=1 + 274)

    args = parser.parse_args()

    model_checkpoint, device = handler.load_model()
    model = default.load_model(model_checkpoint, device)

    print('Using device:', device)

    auc, mrr, ndcg5, ndcg10 = evaluate(args, model, '../data/test_mind_large',
                                       args.num_workers)
    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    )