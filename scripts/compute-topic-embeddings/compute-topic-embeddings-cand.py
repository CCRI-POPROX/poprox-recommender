import json

import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim

from poprox_concepts import CandidateSet
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from poprox_recommender.paths import model_file_path, project_root


def set_seed(seed=42):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


set_seed(42)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
    base_request_data = json.load(req_file)

candidate_article = base_request_data["candidates"]["articles"]

article_topic_map = {}

###make it dataframe
for article in candidate_article:
    article_topics = list(
        {
            mention["entity"]["name"]
            for mention in article["mentions"]
            if mention["entity"]["name"] in TOPIC_DESCRIPTIONS.keys()
        }
    )
    article_topic_map[article["article_id"]] = article_topics


rows = []
index = []
for article_id, topics in article_topic_map.items():
    row = {topic: 1 if topic in topics else 0 for topic in TOPIC_DESCRIPTIONS.keys()}
    rows.append(row)
    index.append(article_id)

article_topic_matrix = pd.DataFrame(rows, index=index)
topic_article_matrix = article_topic_matrix.T

article_embedder = NRMSArticleEmbedder(
    model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=str(device)
)
article_embedding_set = article_embedder(CandidateSet(articles=candidate_article))


M = th.tensor(topic_article_matrix.values, dtype=th.float32, device=device)
num_topics, num_articles = M.shape

embed_dim = 768

topic_emb = nn.Parameter(th.randn(num_topics, embed_dim, device=device))
article_emb = nn.Parameter(article_embedding_set.embeddings.clone())
article_emb.requires_grad = False

# breakpoint()

optimizer = optim.Adam([topic_emb, article_emb], lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

n_epochs = 500
for epoch in range(n_epochs):
    pred = th.matmul(topic_emb, article_emb.T)
    loss = loss_fn(pred, M)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(f'epoch {epoch}: {loss:.5f}')

topic_names = list(topic_article_matrix.index)
topic_embeddings_by_name: dict[str, th.Tensor] = {topic: topic_emb[i].detach() for i, topic in enumerate(topic_names)}

article_ids = list(article_topic_matrix.index)
article_embeddings_by_id: dict[str, th.Tensor] = {id: article_emb[i].detach() for i, id in enumerate(article_ids)}

Tech = topic_embeddings_by_name["Health"]
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
