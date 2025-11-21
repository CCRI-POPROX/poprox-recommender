-- SQL code to import MIND data into DuckDB.
-- The path to the MIND data should be in the SQL variable 'mind_path'.
-- Default is the current directory.

--- create V7 UUIDs for impressions, based on the timestamp *from the
--- impression log*.  The remaining "random" bits are taken from the
--- UUIDV5 computed from the impression ID.  This gives us deterministic
--- impression UUIDs that are still sorted.
CREATE MACRO make_imp_uuid(imp_id, imp_ts) AS (
    (sha_uuid('https://data.poprox.io/mind/impression/', imp_id::VARCHAR)::BYTEA::BITSTRING
    -- 00000000-0000-0FFF-FFFF-FFFFFFFFFFFF strips down to random + variant
    & unhex('0000000000000FFFFFFFFFFFFFFFFFFF')::BITSTRING)
    -- add the UUID version (7)
    | unhex('00000000000070000000000000000000')::BITSTRING
    -- add the 48-bit ms timestamp
    | (bitstring(epoch_ms(imp_ts)::bitstring, 128) << 80)
)::BYTEA::UUID;


CREATE TEMPORARY VIEW raw_behaviors AS
SELECT * FROM read_csv(
    COALESCE(getvariable('mind_path'), '.') || '/behaviors.tsv',
    delim='\t',
    columns={
        'impression_id': 'INT',
        'user_id': 'VARCHAR',
        'time': 'VARCHAR',
        'clicked_news': 'VARCHAR',
        'impressed_news': 'VARCHAR',
    }
);

CREATE TEMPORARY VIEW raw_news AS
SELECT * FROM read_csv(
    COALESCE(getvariable('mind_path'), '.') || '/news.tsv',
    delim='\t',
    columns={
        'article_id': 'VARCHAR',
        'category': 'VARCHAR',
        'subcategory': 'VARCHAR',
        'title': 'VARCHAR',
        'abstract': 'VARCHAR',
        'url': 'VARCHAR',
        'title_entities': 'VARCHAR',
        'abstract_entities': 'VARCHAR',
    }
);

-- Ingest the news articles, generating UUIDs.
TRUNCATE articles;
INSERT INTO articles
SELECT
    -- remove leading char to yield a number
    CAST(article_id[2:] AS INTEGER) AS aid,
    sha_uuid('https://data.poprox.io/mind/article/', article_id),
    category,
    subcategory,
    title,
    abstract,
    title_entities,
    abstract_entities,
FROM raw_news
ORDER BY aid;

-- Ingest the impressions with partial parsing to make tables.
CREATE TEMPORARY TABLE parsed_impressions AS
SELECT
    impression_id AS imp_id,
    make_imp_uuid(impression_id, strptime(time, '%m/%d/%Y %H:%M:%S %p')) AS imp_uuid,
    CAST(user_id[2:] AS INTEGER) user_id,
    strptime(time, '%m/%d/%Y %H:%M:%S %p') AS imp_time,
    string_split_regex(clicked_news, '\s+') AS clicked_news,
    string_split_regex(impressed_news, '\s+') AS impressed_news,
FROM raw_behaviors;

-- Create the impressions table from the partially-parsed table.
TRUNCATE impressions;
INSERT INTO impressions
SELECT imp_id, imp_uuid, user_id, imp_time,
    CAST(julian(imp_time) AS INTEGER)
FROM parsed_impressions
ORDER BY imp_uuid;

-- Extract the users' historical articles.
TRUNCATE impression_history;
INSERT INTO impression_history
SELECT imp_id, CAST(article_id[2:] AS INTEGER)
FROM (
    SELECT imp_id, UNNEST(clicked_news) AS article_id
    FROM parsed_impressions
)
ORDER BY imp_id;

-- Extract the impressed articles.
TRUNCATE impression_articles;
INSERT INTO impression_articles
SELECT imp_id,
    CAST(regexp_extract(article, 'N(\d+)-\d$', 1) AS INTEGER),
    NULLIF(regexp_extract(article, '-(\d)$', 1), '') = '1'
FROM (
    SELECT imp_id, UNNEST(impressed_news) AS article
    FROM parsed_impressions
)
ORDER BY imp_id;

-- Compute summary statistics from impressed articles
TRUNCATE impression_article_summaries;
INSERT INTO impression_article_summaries
SELECT article_id,
    MIN(imp_time), MAX(imp_time),
    MIN(imp_day), MAX(imp_day),
    COUNT(imp_id), SUM(clicked)
FROM impressions
JOIN impression_articles USING (imp_id)
GROUP BY article_id
ORDER BY MIN(imp_day);
