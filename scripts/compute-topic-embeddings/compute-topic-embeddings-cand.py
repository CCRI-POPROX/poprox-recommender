import json
import pandas as pd
import torch as th
from poprox_recommender.paths import project_root
from sklearn.decomposition import TruncatedSVD
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from safetensors.torch import save_file


with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
    base_request_data = json.load(req_file)
    
candidate_article = base_request_data["candidates"]["articles"]

article_topic_map = {}  

for article in candidate_article:
    article_topics = list({mention["entity"]["name"] for mention in article["mentions"] if mention["entity"]["name"] in TOPIC_DESCRIPTIONS.keys()})
    article_topic_map[article["article_id"]] = article_topics
    

rows = []
index = []
for article_id, topics in article_topic_map.items():
    row = {topic: 1 if topic in topics else 0 for topic in TOPIC_DESCRIPTIONS.keys()}
    rows.append(row)
    index.append(article_id)

article_topic_matrix = pd.DataFrame(rows, index=index)
topic_article_matrix = article_topic_matrix.T

svd = TruncatedSVD(n_components=topic_article_matrix.shape[1], random_state=42)
topic_embedding_matrix = svd.fit_transform(topic_article_matrix)

topic_embeddings_by_name: dict[str, th.Tensor] = {
    topic: th.tensor(embedding, dtype=th.float32)
    for topic, embedding in zip(topic_article_matrix.index, topic_embedding_matrix)
}

breakpoint()
    
    

    

