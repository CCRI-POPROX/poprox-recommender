--- Database table definitions for POPROX ingest.
CREATE TABLE newsletters (
    newsletter_id UUID NOT NULL PRIMARY KEY,
    account_id UUID NOT NULL,
    created_at TIMESTAMP NOT NULL,
    recommender_name VARCHAR,
    recommender_version VARCHAR,
    recommender_hash VARCHAR,
    newsletter_date DATE GENERATED ALWAYS AS (created_at::DATE)
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
