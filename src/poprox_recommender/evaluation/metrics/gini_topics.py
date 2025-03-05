from collections import Counter

import numpy as np
from lenskit.stats import gini

from poprox_concepts.domain import CandidateSet
from poprox_recommender.topics import extract_general_topics


def gini_coeff(final_recs: CandidateSet) -> float:
    topic_counter = Counter()
    for article in final_recs.articles:
        topic_counter.update(extract_general_topics(article))

    topic_counts = np.array(list(topic_counter.values()))

    if topic_counts.size == 0 or topic_counts.size == 1:
        return 0.0

    return gini(topic_counts)
