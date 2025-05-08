# PORPOX data sharing
# NOTE TO REVIEWERS: carefully check that any new lines are setting
# appropriate sharing permissions.

# there are 2 sharing commands: 'public' and 'shared'
public models/*-mind
public outputs/mind-*/*/*.parquet outputs/mind-*/*/*.csv.gz outputs/*/*/*-task.json
shared -R outputs/mind-*/
public -R tests/request_data
