-- Pre-compute candidate lists for newsletter dates.
-- Can be replaced later with a table from updated export.
CREATE TABLE newsletter_candidates AS
SELECT DISTINCT a.created_at::DATE AS newsletter_date, article_id
FROM candidate_articles a;
