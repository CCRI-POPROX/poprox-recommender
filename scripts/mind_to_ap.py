##TODO::
# 1. extract AP corpass
# 2. save embeddings of AP article in a safetensor form
#       a. pass the headline of article through news encoder to get the embedings
# 3. save embeddings of Mind article in a safetensor form
#       a. pass the headline of article through news encoder to get the embedings
# 4. for each Mind article,
#       a. calculate it's distance with all the AP articles
#       b. determine neighbors
#       c. assign topics
#       d. store the augmented mind data (question where??)


# how to assign topics???
# 1. union (assign all the topics form all the articles)
# 2. intersection
# 3. union with support (if multiple articles (>K) have that topic)
# 4. union with weighted support (how close the article to the topic or frequency)
# 5. curate the corpass with numbers topics
