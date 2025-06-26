import json

import pandas as pd

from poprox_recommender.paths import project_root

data = project_root() / "data"
rec_df = pd.read_parquet(data / "Test" / "recommendation.parquet")

for id, row in rec_df.iterrows():
    response = json.loads(row.response)
    topic_name = row.persona_name
    total_topical_candidate = row.candidate_count
    articles = response["recommendations"]["articles"]
    scores = response["recommendations"]["scores"]
    topical_count = 0
    for i, (article, score) in enumerate(zip(articles, scores)):
        article_topics = set([m["entity"]["name"] for m in article["mentions"] if m["entity"] is not None])
        if topic_name in article_topics:
            topical_count += 1

    recall = topical_count/total_topical_candidate if total_topical_candidate else float('nan')
    precision = topical_count/10

    print(
        f"{topic_name}: precision: {precision} recall: {recall}"
    )
    for i, (article, score) in enumerate(zip(articles, scores)):
        article_topics = set([m["entity"]["name"] for m in article["mentions"] if m["entity"] is not None])
        print(f"{i+1}. [{score:.4f}] {article['headline']} {article_topics}")
