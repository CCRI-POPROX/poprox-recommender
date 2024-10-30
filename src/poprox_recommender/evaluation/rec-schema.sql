-- SQL schema for offline recommendation database

-- list metadata — user, recommender etc.
-- we're using varchar for the categoricals, duckdb should use dictionary encoding
CREATE TABLE rec_list_meta (
    rl_id INTEGER PRIMARY KEY,
    recommender VARCHAR NOT NULL,
    user_id UUID,
    stage VARCHAR NOT NULL,
);

-- individual articles in a recommendation list
CREATE TABLE rec_list_articles (
    rl_id INTEGER NOT NULL,
    rank INT16 NOT NULL,
    article_id UUID NOT NULL,
    score FLOAT NULL,
    embedding FLOAT[] NULL,
);

-- LensKit-compatible view of final recommendation data
CREATE VIEW lk_recommendations AS
SELECT rl_id, recommender, user_id AS user, article_id AS item, rank, score
FROM rec_list_articles
JOIN rec_list_meta USING (rl_id)
-- I would like to order by rank within user_id to reduce overall sorting,
-- but SQL does not allow that (so far as I know); this should be mostly
-- sorted already, so fast. — ME
ORDER BY rl_id, rank;
