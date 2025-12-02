-- Pre-compute candidate lists for newsletter dates.
-- Can be replaced later with a table from updated export.
-- Using dates to locate candidates is fine, since we are running in the morning
-- Eastern time — newsletter and article mismatch would only be a problem if we
-- are still running the ingeste and newsletter jobs after 6pm Eastern.
CREATE TABLE newsletter_candidates AS
SELECT DISTINCT a.created_at::DATE AS newsletter_date, article_id
FROM candidate_articles a;
