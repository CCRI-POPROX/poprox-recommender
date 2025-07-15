import json

import pandas as pd

from poprox_recommender.paths import project_root

data = project_root() / "data"
rec_df = pd.read_parquet(data / "Test" / "recommendation" / "nrms.parquet")

row = rec_df.iloc[0]

response = json.loads(row["response"])
topic_name = row["persona_name"]

articles = response["recommendations"]["articles"]
scores = response["recommendations"]["scores"]

print(topic_name)
for i, (article, score) in enumerate(zip(articles, scores)):
    article_topics = set([m["entity"]["name"] for m in article["mentions"] if m["entity"] is not None])
    print(f"{i+1}. [{score:.4f}] {article['headline']} {article_topics}")
