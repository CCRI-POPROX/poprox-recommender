from ast import literal_eval

import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, behaviors_path, news_path):
        super(BaseDataset, self).__init__()

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col="id",
            usecols=["id"] + [args.dataset_attributes],
            converters={attribute: literal_eval for attribute in [args.dataset_attributes]},
        )

        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        # {'news ID': {'title': [list of token]}}
        self.news2dict = self.news_parsed.to_dict("index")

        self.args = args
        # convert token list to tensor
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2])

        padding_all = {
            "title": [0] * self.args.num_words_title,
        }

        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        # if empty, padding title token by zeros
        self.padding = {k: v for k, v in padding_all.items() if k in [args.dataset_attributes]}

        if self.args.subset is not None:
            self.behaviors_parsed = self.behaviors_parsed.iloc[: self.args.subset]

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        """
        {'clicked': [1, 0, 0, 0, 0],
         'candidate_news': [{'title': tensor([  101,  1302,   119,   122,  6831,  1419, 20651, 11102, 15872,  3968,
                    1107,  4555,   102,     0,     0,     0,     0,     0,     0,     0])},
          {'title': tensor([ 101, 4872, 1104, 3764, 2631, 1873, 4601,  132, 1769, 2606, 1276,  102,
                      0,    0,    0,    0,    0,    0,    0,    0])},
          {'title': tensor([  101,  4238,   152, 26358,  3276,   112, 11451,  3537,  1228,   170,
                    2727,   112,  1104,  1123,  5656, 25265,  1165,  1131,  2204,   102])},
          {'title': tensor([  101,  4369,   112,   188, 26700, 14623,   117,  6197,  1663,   159,
                     117,  3370,  1103, 10909,  1111,  1217,  1315,  2698,   102,     0])},
          {'title': tensor([  101,   112,  6356,  3259,  2289,   112, 12895,  1223,  1783,  1170,
                    1129, 12888, 13756,  5879,  1104, 19531, 27219,  1111,  4007,   102])}],
         'clicked_news': [
         {'title': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
         {'title': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
         {'title': tensor([  101,   112, 11679,  9208,  4746,   188,  2430,  3624,  2787,   112,
                    3950,  1254,  1120,   152,   112, 18146,   117,  2290,  2021,   102])},
         {'title': tensor([  101, 12646, 14926,  1116,  1111,  7589,  3764,  1107,  1203,  5705,
                    9322,  2977,  4556,  1459,  1751,  7546,  1115,  1841,   123,   102])}]}
        """
        item = {}
        row = self.behaviors_parsed.iloc[idx]

        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [self.news2dict[x] for x in row.candidate_news.split()]
        item["clicked_news"] = [
            self.news2dict[x] for x in row.clicked_news.split()[: self.args.num_clicked_news_a_user]
        ]

        repeated_times = self.args.num_clicked_news_a_user - len(item["clicked_news"])

        assert repeated_times >= 0
        item["clicked_news"] = [self.padding] * repeated_times + item["clicked_news"]

        sampled_item = {}
        sampled_item["candidate_news"] = torch.stack([t["title"] for t in item["candidate_news"]])  # shape = [5, 20]
        sampled_item["clicked_news"] = torch.stack([t["title"] for t in item["clicked_news"]])  # shape = [4, 20]
        sampled_item["clicked"] = torch.tensor(0)

        return sampled_item


class ValDataset(Dataset):
    def __init__(self, args, behaviors_path, news_path):
        super(ValDataset, self).__init__()

        self.behaviors = pd.read_table(
            behaviors_path,
            header="infer",
            usecols=range(5),
            names=["impression_id", "user", "time", "clicked_news", "impressions"],
        )
        self.behaviors.fillna({"clicked_news": " "}, inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()
        self.behaviors["clicked"] = self.behaviors.impressions.apply(lambda x: [item.split("-")[1] for item in x])
        self.behaviors["candidate_news"] = self.behaviors.impressions.apply(
            lambda x: [item.split("-")[0] for item in x]
        )

        self.news_parsed = pd.read_table(
            news_path,
            index_col="id",
            usecols=["id"] + [args.dataset_attributes],
            converters={attribute: literal_eval for attribute in [args.dataset_attributes]},
        )

        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        # {'news ID': {'title': [list of token]}}
        self.news2dict = self.news_parsed.to_dict("index")

        self.args = args
        # convert token list to tensor
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2])

        padding_all = {
            "title": [0] * self.args.num_words_title,
        }

        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        # if empty, padding title token by zeros
        self.padding = {k: v for k, v in padding_all.items() if k in [args.dataset_attributes]}

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors.iloc[idx]

        item["clicked"] = list(map(int, row.clicked))  # labels (list)
        item["candidate_news"] = [self.news2dict[x] for x in row.candidate_news]

        item["clicked_news"] = [
            self.news2dict[x] for x in row.clicked_news.split()[: self.args.num_clicked_news_a_user]
        ]

        repeated_times = self.args.num_clicked_news_a_user - len(item["clicked_news"])

        assert repeated_times >= 0
        item["clicked_news"] = [self.padding] * repeated_times + item["clicked_news"]

        sampled_item = {}
        sampled_item["candidate_news"] = torch.stack([t["title"] for t in item["candidate_news"]])  # shape = [5, 20]
        sampled_item["clicked_news"] = torch.stack([t["title"] for t in item["clicked_news"]])  # shape = [4, 20]
        sampled_item["clicked"] = torch.tensor(0)

        return sampled_item
