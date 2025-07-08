from single_tag_vs_multi_tag import multi_topical_articles, single_topical_articles
from testArticleEmb import embed_article

if __name__ == "__main__":
    for topic, articles in single_topical_articles.items():
        for article in articles:
            article_embedding = embed_article(article["headline"])
            breakpoint()
