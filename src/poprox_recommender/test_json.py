import io
import json
import os
import sys
from lenskit.metrics import topn
sys.path.append('src')
import torch as th
from safetensors.torch import load_file
from uuid import UUID
import numpy as np
import pandas as pd
from tqdm import tqdm
from poprox_concepts import Article, ClickHistory
from poprox_recommender.default import select_articles
from poprox_recommender.paths import project_root



def load_model(device_name=None):

    if device_name is None:
        device_name = "cuda" if th.cuda.is_available() else "cpu"

    load_path = f"{project_root()}/models/model.safetensors"
    checkpoint = load_file(load_path)

    return checkpoint, device_name


def custom_encoder(obj):
    if isinstance(obj, UUID):
        return str(obj)

def compute_mrr(y_pred, y_true):
    # 关心原本label为1的item在y_pred中的位置 y_pred =list[排好的item id]
    relevant_items = {item.split('-')[0] for item in y_true if item.endswith('-1')}
    mrr = 0
    for rank, pred_item in enumerate(y_pred, start = 1):
        if pred_item in relevant_items:
            mrr += (1/rank)
    return mrr/len(relevant_items)



def compute_dcg(y_pred, y_true, k=5):
    '''
    y_pred: 排好的item顺序
    y_true:[item-label]
    '''
    item_label = {}
    for pair in y_true:
        item_label[pair.split('-')[0]] = int(pair.split('-')[1])

    y_true = np.array([item_label[item] for item in y_pred[:k]])

    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)

    return np.sum(gains / discounts)


def compute_ndcg(y_pred, y_true, k):
    '''
    y_pred: 排好的item顺序
    y_true:[item-label]
    '''
    # y_true2 作为best的item排序
    y_true_score = [item.split('-')[1] for item in y_true]
    order = np.argsort(y_true_score)[::-1]
    y_true_item = [y_true[index].split('-')[0] for index in order]

    best = compute_dcg(y_true_item, y_true, k)
    actual = compute_dcg(y_pred, y_true, k)
    return actual / best

def recsys_metric(recommendations, row_index, news_struuid_ID ):
    # recommendations {account id (uuid): LIST[Article]}
    # use the url of Article
    impressions_truth = pd.read_table(f"{project_root()}/data/test_mind_large/behaviors.tsv",
                                    header='infer',
                                    usecols=range(5),
                                    names=[
                                        'impression_id', 'user', 'time',
                                        'clicked_news', 'impressions'
                                    ]).iloc[row_index].impressions.split() # LIST[ID-label]

    account_id = list(recommendations.keys())[0]
    recommended_list = recommendations[account_id]
    recommended_list = [news_struuid_ID[item.url] for item in recommended_list]
    
    
    recs = pd.DataFrame({
        'item': recommended_list
    })
    
    truth = pd.DataFrame.from_records(
        (
            (row.split('-')[0], int(row.split('-')[1]))
            for row in impressions_truth
        ),
        columns=['item', 'rating']
    ).set_index('item')

    single_rr = topn.recip_rank(recs, truth)
    single_ndcg5 = topn.ndcg(recs, truth, k=5)
    single_ndcg10 = topn.ndcg(recs, truth, k=10)

    # single_rr = compute_mrr(recommended_list, impressions_truth)
    # single_ndcg5 = compute_ndcg(recommended_list, impressions_truth, 5)
    # single_ndcg10 = compute_ndcg(recommended_list, impressions_truth, 10)

    return single_ndcg5, single_ndcg10, single_rr

if __name__ == '__main__':
    '''
    For offline evaluation, set theta in mmr_diversity = 1
    '''
    MODEL, DEVICE = load_model()
    TOKEN_MAPPING = 'distilbert-base-uncased'  # can be modified

    with open(f"{project_root()}/data/val_mind_large/news_uuid_ID.json", 'r') as json_file:
        news_struuid_ID = json.load(json_file)

    # load the mind test json file
    with open(f"{project_root()}/data/val_mind_large/mind_test.json", 'r') as json_file:
        mind_data = json.load(json_file)

    ndcg5 = []
    ndcg10 = []
    mrr = []

    for impression_idx in range(10): # one by one

        request_body = mind_data[impression_idx]

        todays_articles = [Article.parse_obj(item) for item in request_body["todays_articles"]]
        past_articles = [Article.parse_obj(item) for item in request_body["past_articles"]]

        click_data = [ClickHistory.parse_obj(request_body["click_data"])]


        num_recs = len(todays_articles)

        recommendations = select_articles(
            todays_articles,
            past_articles,
            click_data,
            MODEL,
            DEVICE,
            TOKEN_MAPPING,
            num_recs,
        )

        single_ndcg5, single_ndcg10, single_mrr = recsys_metric(recommendations, impression_idx, news_struuid_ID )
        # recommendations {account id (uuid): LIST[Article]}
        print(f"----------------evaluation using the first {impression_idx + 1} is NDCG@5 = {single_ndcg5}, NDCG@10 = {single_ndcg10}, RR = {single_mrr}")

        ndcg5.append(single_ndcg5)
        ndcg10.append(single_ndcg10)
        mrr.append(single_mrr)



    print(f"Offline evaluation metrics on MIND data: NDCG@5 = {np.mean(ndcg5)}, NDCG@10 = {np.mean(ndcg10)}, MRR = {np.mean(mrr)}")


        #response = {"statusCode": 200, "body": json.dump(body, default=custom_encoder)}




