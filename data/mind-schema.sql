-- Schema for representing MIND data in DuckDB.

-- Table for article data from news.tsv
CREATE TABLE articles (
    article_id INT NOT NULL PRIMARY KEY,
    article_uuid UUID NOT NULL UNIQUE,
    category VARCHAR,
    subcategory VARCHAR,
    title VARCHAR,
    abstract VARCHAR,
    title_entities JSON,
    abstract_entities JSON,
);

-- Table for impressions and behavior logs from behaviors.tsv.
CREATE TABLE impressions (
    imp_id INT NOT NULL PRIMARY KEY,
    imp_uuid UUID NOT NULL UNIQUE,
    user_id INT NOT NULL,
    imp_time TIMESTAMP NOT NULL,
    -- Julian day number for each impression to put in days.
    imp_day INTEGER NOT NULL,
);

-- The previously-clicked articles in an impression record (user history).
CREATE TABLE impression_history (
    imp_id INT NOT NULL,
    article_id INT NOT NULL,
);
CREATE INDEX impression_history_idx ON impression_history (imp_id);

-- The impressed ariticles in an impression record.
CREATE TABLE impression_articles (
    imp_id INT NOT NULL,
    article_id INT NOT NULL,
    clicked BOOLEAN NOT NULL,
);
CREATE INDEX impression_articles_imp_idx ON impression_articles (imp_id);

-- Summary information about an article's appearance in impressions.
-- Making this a table since DuckDB does not support materialized views.
CREATE TABLE impression_article_summaries (
    article_id INT NOT NULL,
    -- First time the article appeared.
    first_time TIMESTAMP NOT NULL,
    -- Last time the article appeared.
    last_time TIMESTAMP NOT NULL,
    -- Julian day number of the first day the article appeared.
    first_day INTEGER NOT NULL,
    -- Julian day number of the last day the article appeared.
    last_day INTEGER NOT NULL,
    -- Number of impressions in which this article appeared.
    num_impressions INTEGER NOT NULL,
    -- Number of times this article was clicked.
    num_clicks INTEGER NOT NULL,
);
CREATE INDEX impression_summary_idx ON impression_article_summaries (article_id);

-- View for expanded candidate sets.
-- DO NOT USE - these are broken
CREATE VIEW impression_expanded_candidates AS
SELECT imp_id, article_id
FROM impressions, impression_article_summaries
WHERE first_day <= imp_day
AND last_day > imp_day - 7
ORDER BY imp_id;
