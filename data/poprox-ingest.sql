-- POPROX database ingest.
-- Extract newsletter metadata
-- EXPORT REQUEST: separate list of newsletters from newsletter contents
INSERT INTO newsletters
SELECT DISTINCT
    newsletter_id, account_id, created_at,
    recommender_name, recommender_version, recommender_hash
FROM 'data/POPROX/newsletters.parquet';

INSERT INTO newsletter_articles
SELECT
    newsletter_id, article_id, position, headline, subhead
FROM 'data/POPROX/newsletters.parquet';
CREATE INDEX newsletter_article_idx ON newsletter_articles (article_id);

INSERT INTO clicks
SELECT account_id, newsletter_id, article_id, clicked_at
FROM 'data/POPROX/clicks.parquet'
ORDER BY account_id, newsletter_id;
CREATE INDEX click_account_idx ON clicks (account_id);
CREATE INDEX click_newsletter_idx ON clicks (newsletter_id);

INSERT INTO interests
SELECT account_id, entity_id, entity_name, preference
FROM 'data/POPROX/interests.parquet'
ORDER BY account_id;

INSERT INTO candidate_articles
SELECT
    article_id, headline, subhead, url,
    linked_articles, raw_data, published_at, created_at, body
FROM 'data/POPROX/articles.parquet'
ORDER BY created_at;

INSERT INTO candidate_article_mentions
SELECT mention_id, article_id, source, relevance, entity
FROM 'data/POPROX/article_mentions.parquet'
ORDER BY article_id;
CREATE INDEX candidate_article_mention_article_idx ON candidate_article_mentions (article_id);

INSERT INTO clicked_articles
SELECT
    article_id, headline, subhead, url,
    linked_articles, raw_data, published_at, created_at
FROM 'data/POPROX/clicked/articles.parquet'
ORDER BY article_id;

INSERT INTO clicked_article_mentions
SELECT mention_id, article_id, source, relevance, entity
FROM 'data/POPROX/clicked/article_mentions.parquet'
ORDER BY article_id;

CREATE INDEX clicked_article_mention_article_idx ON clicked_article_mentions (article_id);
