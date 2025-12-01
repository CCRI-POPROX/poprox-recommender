CREATE TABLE newsletters (
    newsletter_id UUID NOT NULL PRIMARY KEY,
    account_id UUID NOT NULL,
    created_at TIMESTAMP NOT NULL,
    recommender_name VARCHAR,
    recommender_version VARCHAR,
    recommender_hash VARCHAR,
);

CREATE TABLE newsletter_articles (
    newsletter_id UUID NOT NULL,
    article_id UUID NOT NULL,
    position SMALLINT NOT NULL,
    headline VARCHAR,
    subhead VARCHAR
);

CREATE TABLE clicks (
    account_id UUID NOT NULL,
    newsletter_id UUID NOT NULL,
    article_id UUID NOT NULL,
    clicked_at TIMESTAMP NOT NULL,
);

CREATE TABLE interests (
    account_id UUID NOT NULL,
    entity_id UUID NOT NULL,
    entity_name VARCHAR NOT NULL,
    preference TINYINT NOT NULL,
);

CREATE TABLE candidate_articles (
    article_id UUID NOT NULL PRIMARY KEY,
    headline VARCHAR,
    subhead VARCHAR,
    url VARCHAR,
    linked_articles VARCHAR,
    raw_data JSON,
    published_at TIMESTAMP,
    created_at TIMESTAMP,
    body VARCHAR,
);

CREATE TABLE candidate_article_mentions (
    mention_id UUID PRIMARY KEY,
    article_id UUID NOT NULL,
    source VARCHAR,
    relevance FLOAT,
    entity JSON,
);

CREATE TABLE clicked_articles (
    article_id UUID NOT NULL PRIMARY KEY,
    headline VARCHAR,
    subhead VARCHAR,
    url VARCHAR,
    linked_articles VARCHAR,
    raw_data JSON,
    published_at TIMESTAMP,
    created_at TIMESTAMP,
);

CREATE TABLE clicked_article_mentions (
    mention_id UUID PRIMARY KEY,
    article_id UUID NOT NULL,
    source VARCHAR,
    relevance FLOAT,
    entity JSON,
);

-- Extract newsletter metadata
-- EXPORT REQUEST: separate list of newsletters from newsletter contents
INSERT INTO newsletters
SELECT DISTINCT
    newsletter_id, account_id, created_at,
    recommender_name, recommender_version, recommender_hash
FROM 'POPROX/newsletters.parquet';

INSERT INTO newsletter_articles
SELECT
    newsletter_id, article_id, position, headline, subhead
FROM 'POPROX/newsletters.parquet';
CREATE INDEX newsletter_article_idx ON newsletter_articles (article_id);

INSERT INTO clicks
SELECT account_id, newsletter_id, article_id, clicked_at
FROM 'POPROX/clicks.parquet'
ORDER BY account_id, newsletter_id;
CREATE INDEX click_account_idx ON clicks (account_id);
CREATE INDEX click_newsletter_idx ON clicks (newsletter_id);

INSERT INTO interests
SELECT account_id, entity_id, entity_name, preference
FROM 'POPROX/interests.parquet'
ORDER BY account_id;

INSERT INTO candidate_articles
SELECT
    article_id, headline, subhead, url,
    linked_articles, raw_data, published_at, created_at, body
FROM 'POPROX/articles.parquet'
ORDER BY created_at;

INSERT INTO candidate_article_mentions
SELECT mention_id, article_id, source, relevance, entity
FROM 'POPROX/article_mentions.parquet'
ORDER BY article_id;
CREATE INDEX candidate_article_mention_article_idx ON candidate_article_mentions (article_id);

INSERT INTO clicked_articles
SELECT
    article_id, headline, subhead, url,
    linked_articles, raw_data, published_at, created_at
FROM 'POPROX/clicked/articles.parquet'
ORDER BY article_id;

INSERT INTO clicked_article_mentions
SELECT mention_id, article_id, source, relevance, entity
FROM 'POPROX/clicked/article_mentions.parquet'
ORDER BY article_id;

CREATE INDEX clicked_article_mention_article_idx ON clicked_article_mentions (article_id);
