import csv
import os
import random
from os import path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from poprox_recommender.paths import project_root


def parse_behaviors(source, target, user2int_path, negative_sampling_ratio):
    """
    parse_behaviors is only for training data
    Input: behavior.tsv
    Output: behaviors_parsed.tsv after negative sampling
            user  int      clicked_news [N... N...]     candidate_news [N... N...]          clicked [1 0 0 0 0]
    """
    behaviors = pd.read_table(
        source, header="infer", names=["impression_id", "user", "time", "clicked_news", "impressions"]
    )

    # during the training, allow the existence of users who have no past interactions
    behaviors.fillna({"clicked_news": " "}, inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    # convert raw user ID to int
    user2int = {}
    for row in behaviors.itertuples(index=False):
        if row.user not in user2int:
            user2int[row.user] = len(user2int) + 1

    pd.DataFrame(user2int.items(), columns=["user", "int"]).to_csv(user2int_path, sep="\t", index=False)

    print("The number of unique users in TRAINING is ", 1 + len(user2int))

    for row in behaviors.itertuples():
        behaviors.at[row.Index, "user"] = user2int[row.user]

    for row in tqdm(behaviors.itertuples(), desc="Negative sampling for training"):
        positive = iter([x for x in row.impressions if x.endswith("1")])
        negative = [x for x in row.impressions if x.endswith("0")]
        random.shuffle(negative)
        negative = iter(negative)
        pairs = []
        try:
            while True:
                pair = [next(positive)]
                for _ in range(negative_sampling_ratio):
                    pair.append(next(negative))
                pairs.append(pair)
        except StopIteration:
            pass
        behaviors.at[row.Index, "impressions"] = pairs

    # drop those rows that do not satisfy 1:4 ratio
    behaviors = behaviors.explode("impressions").dropna(subset=["impressions"]).reset_index(drop=True)

    behaviors[["candidate_news", "clicked"]] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (" ".join([e.split("-")[0] for e in x]), " ".join([e.split("-")[1] for e in x]))
        ).tolist()
    )

    # save the parsed training behavior
    behaviors.to_csv(target, sep="\t", index=False, columns=["user", "clicked_news", "candidate_news", "clicked"])


def parse_news(source, target, pretrained_tokenizer, token_length):
    """
    Parse_news is applied to training, validation, and testing data
    Input: news.tsv
    Output: news_parsed.tsv
            id (N...)         title [list of tokens]
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    news = pd.read_table(
        source,
        header="infer",
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=["id", "category", "subcategory", "title", "abstract", "title_entities", "abstract_entities"],
    )

    news.fillna(" ", inplace=True)

    def parse_row(text):
        # for NRMS_bert, we only encode 'title' so far
        return tokenizer.encode(text, padding="max_length", max_length=token_length, truncation=True)

    news["title"] = news["title"].apply(lambda x: parse_row(x))

    news.to_csv(target, sep="\t", index=False)


if __name__ == "__main__":
    """
    The followings need modification based on need
    """
    root = project_root()
    train_dir = root / "data/MINDlarge_train"
    val_dir = root / "data/MINDlarge_dev"

    post_train_dir = root / "data/MINDlarge_post_train"
    post_val_dir = root / "data/MINDlarge_post_dev"

    os.makedirs(post_train_dir, exist_ok=True)
    os.makedirs(post_val_dir, exist_ok=True)

    pretrain_tokenizer = (
        "distilbert-base-uncased"  # the model needs to be consistent with the pretrained model during training
    )
    max_length = 30
    negative_sampling_ratio = 4

    """
    Preprocess data
    """
    print("Parse behaviors and news for Training")
    parse_behaviors(
        path.join(train_dir, "behaviors.tsv"),
        path.join(post_train_dir, "behaviors_parsed.tsv"),
        path.join(post_train_dir, "user2int.tsv"),
        negative_sampling_ratio,
    )

    parse_news(
        path.join(train_dir, "news.tsv"),
        path.join(post_train_dir, "news_parsed.tsv"),
        pretrained_tokenizer=pretrain_tokenizer,
        token_length=max_length,
    )

    print("\nProcess news for Validation")
    parse_news(
        path.join(val_dir, "news.tsv"),
        path.join(post_val_dir, "news_parsed.tsv"),
        pretrained_tokenizer=pretrain_tokenizer,
        token_length=max_length,
    )
