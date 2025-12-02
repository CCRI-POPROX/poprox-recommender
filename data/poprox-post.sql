-- Pre-compute candidate lists for newsletter dates.
-- Can be replaced later with a table from updated export.
CREATE TABLE newsletter_candidates AS
SELECT DISTINCT newsletter_date, article_id
FROM newsletters n, candidate_articles a
WHERE n.created_at::DATE = a.created_at::DATE;
