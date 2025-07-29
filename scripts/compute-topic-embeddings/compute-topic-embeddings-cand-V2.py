import json
import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import save_file

from poprox_concepts import CandidateSet
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from poprox_recommender.paths import model_file_path, project_root


def set_seed(seed=42):
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


set_seed(42)


device = th.device("cuda" if th.cuda.is_available() else "cpu")


with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
    base_request_data = json.load(req_file)

candidate_article = base_request_data["candidates"]["articles"]

topic_names = set()

for article in candidate_article:
    for mention in article.get("mentions", []):
        entity = mention.get("entity")
        topic_names.add(entity["name"])
topic_to_idx = {t: i for i, t in enumerate(topic_names)}


article_ids = []
article_to_idx = {}

pos_t_idx = []
pos_a_idx = []

for article in candidate_article:
    article_id = article["article_id"]
    if article_id not in article_to_idx:
        article_to_idx[article_id] = len(article_ids)
        article_ids.append(article_id)
    a_idx = article_to_idx[article_id]

    topics = set()
    for mention in article.get("mentions", []):
        entity = mention.get("entity")
        topic = entity["name"]
        if topic in topic_to_idx:
            t_idx = topic_to_idx[topic]
            pos_t_idx.append(t_idx)
            pos_a_idx.append(a_idx)

num_topics = len(topic_names)
num_articles = len(article_ids)
embed_dim = 768


M = th.zeros((num_topics, num_articles), dtype=th.float32, device=device)
M[pos_t_idx, pos_a_idx] = 1.0

article_embedder = NRMSArticleEmbedder(
    model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=str(device)
)
article_embedding_set = article_embedder(CandidateSet(articles=candidate_article))

topic_emb = nn.Parameter(th.randn(num_topics, embed_dim, device=device))
article_emb = nn.Parameter(article_embedding_set.embeddings.clone())
article_emb.requires_grad = False


optimizer = optim.Adam([topic_emb], lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

n_epochs = 500
for epoch in range(n_epochs):
    pred = th.matmul(topic_emb, article_emb.T)
    loss = loss_fn(pred, M)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


topic_embeddings_by_name: dict[str, th.Tensor] = {topic: topic_emb[i].detach() for topic, i in topic_to_idx.items()}
article_embeddings_by_id: dict[str, th.Tensor] = {aid: article_emb[i].detach() for aid, i in article_to_idx.items()}


Tech = topic_embeddings_by_name["Climate and environment"]
tech_news_rec = {}
for article_id, embedding in article_embeddings_by_id.items():
    score = th.dot(Tech, embedding)
    for cand_art in candidate_article:
        if cand_art["article_id"] == article_id:
            news_topics = list(
                {
                    mention["entity"]["name"]
                    for mention in cand_art["mentions"]
                    if mention["entity"]["name"] in TOPIC_DESCRIPTIONS.keys()
                }
            )
            tech_news_rec[cand_art["headline"]] = [score, news_topics]

sorted_tech_news = dict(sorted(tech_news_rec.items(), key=lambda item: item[1][0], reverse=True))

for i, (headline, (score, topics)) in enumerate(list(sorted_tech_news.items())[:10], 1):
    print(f"{i}. {score:.4f} — {headline}")
    print(f"    Topics: {topics}")
