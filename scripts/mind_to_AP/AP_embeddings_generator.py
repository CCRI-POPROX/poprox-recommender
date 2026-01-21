import json

import pandas as pd
import torch as th
from tqdm import tqdm

from poprox_concepts import Article, CandidateSet, Entity, Mention
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.paths import model_file_path, project_root


def complete_article_generator(row, mentions_df):
    article_mentions_df = mentions_df[mentions_df["article_id"] == row.article_id]
    mentions = [
        Mention(
            mention_id=m_row.mention_id,
            article_id=m_row.article_id,
            source=m_row.source,
            relevance=m_row.relevance,
            entity=Entity(**json.loads(m_row.entity)) if m_row.entity else None,
        )
        for m_row in article_mentions_df.itertuples()
    ]
    article = Article(
        article_id=row.article_id,
        headline=row.headline,
        subhead=row.subhead,
        body=row.body,
        published_at=row.published_at,
        mentions=mentions,
        source="AP",
        external_id="",
        raw_data=json.loads(row.raw_data) if row.raw_data else None,
    )
    return article


# defining device for article_embedder
device = th.device("cuda" if th.cuda.is_available() else "cpu")


# data read
data = project_root() / "data" / "test"
articles_df = pd.read_parquet(data / "articles.parquet")
mentions_df = pd.read_parquet(data / "mentions.parquet")
all_dates = sorted(articles_df["published_at"].dt.normalize().unique())


# taking a tiny set for testing
sample_dates = all_dates[-15:]
sample_df = articles_df[articles_df["published_at"].dt.normalize().isin(sample_dates)]


# turning into complete article that our article_embedder can take as input
# here, each row means each article of the dataframe
candidate_article_obj = []
for row in tqdm(sample_df.itertuples()):
    article = complete_article_generator(row, mentions_df)
    candidate_article_obj.append(article)
candidate_article = [a.model_dump(mode="json", exclude_none=True) for a in candidate_article_obj]


# creating article_index according to position to find the article embedding for each article
# as article_embedder returns it as a batch without any assigned index
article_ids = []
article_to_idx = {}
for article in candidate_article:
    article_id = article["article_id"]
    if article_id not in article_to_idx:
        article_to_idx[article_id] = len(article_ids)
        article_ids.append(article_id)


# declaring NRMS article_embedder & passing articles as a batch
article_embedder = NRMSArticleEmbedder(
    model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=str(device)
)
article_embedding_set = article_embedder(CandidateSet(articles=candidate_article))


# by default NRMS article_embedder append the embeddings at the end of provided article batch
# but we only need the embeddings
article_emb = article_embedding_set.embeddings.clone()


# storing article_id, headline, topic_name and embeddings
# here we are storing all the topics (not only the onboardings)
ap_emb = []
for article in candidate_article:
    aid = article["article_id"]
    headline = article["headline"]
    topics = set()

    for mention in article.get("mentions", []):
        entity = mention.get("entity")
        if entity is not None:
            topic = entity.get("name")
            if topic:
                topics.add(topic)

    idx = article_to_idx[aid]
    ap_emb.append(
        {
            "article_id": aid,
            "headline": headline,
            "topic_name": list(topics),
            "embedding": article_emb[idx].detach().cpu().numpy().tolist(),
        }
    )


# data store
df = pd.DataFrame(ap_emb)
output = project_root() / "models" / "precalculated_model" / "ap_emb.parquet"
df.to_parquet(output, index=False)
