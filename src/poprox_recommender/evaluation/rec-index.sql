--- SQL script to run *after* ingest to set up indexes.
CREATE INDEX rec_list_user_idx ON rec_list_meta (user_id);
CREATE INDEX rec_list_articles_idx ON rec_list_articles (rl_id);
