from uuid import uuid4

import pandas as pd
import torch as th
from tqdm import tqdm

from poprox_concepts import Article, CandidateSet
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.paths import model_file_path, project_root


def minimal_article_generator(row):
    article = Article(
        article_id=uuid4(),
        headline=row.headline,
        # mind article_id is not uuid so storing it as external_id later will restore it as article_id
        external_id=row.article_id,
        source="MIND",
    )
    return article


# defining device for article_embedder
device = th.device("cuda" if th.cuda.is_available() else "cpu")


# data read
data = project_root() / "data" / "MINDsmall_dev" / "news.tsv"
articles_df = pd.read_table(
    data,
    header=None,
    names=[
        "article_id",
        "category",
        "subcategory",
        "headline",
        "abstract",
        "url",
        "headline_entities",
        "abstract_entities",
    ],
)


# taking a tiny set for testing
sample_df = articles_df[-50:]


# turning into an article with minimum possible fields that our article_embedder can take as input
# here, each row means each article of the dataframe
candidate_article_obj = []
for row in tqdm(sample_df.itertuples()):
    article = minimal_article_generator(row)
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


# storing article_id, headline and embeddings
mind_emb = []
for article in candidate_article:
    aid = article["article_id"]
    mind_article_id = article["external_id"]
    headline = article["headline"]

    idx = article_to_idx[aid]
    mind_emb.append(
        {
            "article_id": mind_article_id,
            "headline": headline,
            "embedding": article_emb[idx].detach().cpu().numpy().tolist(),
        }
    )


# data store
df = pd.DataFrame(mind_emb)
output = project_root() / "models" / "precalculated_model" / "mind_emb.parquet"
df.to_parquet(output, index=False)
