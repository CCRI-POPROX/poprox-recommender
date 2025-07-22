from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.components.joiners.score import ScoreFusion, ScoreFusionConfig


def test_generic():
    article = Article(headline="")
    article_list = [article]
    score_list = [0.1]

    n_scorer = CandidateSet(articles=article_list, scores=score_list)
    n_scorer2 = CandidateSet(articles=article_list, scores=score_list)

    fusion = ScoreFusion()
    output = fusion(candidates1=n_scorer, candidates2=n_scorer2)

    assert len(output.articles) == 1
    assert output.scores == [0.2]


def test_epmty_articleset():
    n_scorer0 = CandidateSet(articles=[], scores=[])
    n_scorer00 = CandidateSet(articles=[], scores=[])

    fusion = ScoreFusion()
    output_0 = fusion(candidates1=n_scorer0, candidates2=n_scorer00)

    assert output_0.scores == []


def test_same_articleset_reverse_order():
    article2 = Article(headline="B")
    article3 = Article(headline="C")

    article_list_2 = [article2, article3]
    score_list_2 = [0.3, 0.5]

    article_list_2r = [article3, article2]
    score_list_2r = [0.5, 0.3]

    n_scorer2 = CandidateSet(articles=article_list_2, scores=score_list_2)
    n_scorer2r = CandidateSet(articles=article_list_2r, scores=score_list_2r)

    fusion = ScoreFusion()
    output_r = fusion(candidates1=n_scorer2, candidates2=n_scorer2r)

    assert output_r.scores == [0.6, 1.0]


def test_no_matched_articleset():
    article1 = Article(headline="A")
    article2 = Article(headline="B")
    article3 = Article(headline="C")
    article4 = Article(headline="A")

    article_list_1 = [article1, article2]
    score_list_1 = [0.2, 0.3]

    article_list_3 = [article3, article4]
    score_list_3 = [0.5, 0.2]

    n_scorer = CandidateSet(articles=article_list_1, scores=score_list_1)
    n_scorer3 = CandidateSet(articles=article_list_3, scores=score_list_3)

    fusion = ScoreFusion()

    output_2 = fusion(candidates1=n_scorer, candidates2=n_scorer3)

    assert output_2.scores == [0.2, 0.3, 0.5, 0.2]


def test_overlapped_articleset():
    article1 = Article(headline="A")
    article2 = Article(headline="B")
    article3 = Article(headline="C")

    article_list_1 = [article1, article2]
    score_list_1 = [0.2, 0.3]

    article_list_2 = [article2, article3]
    score_list_2 = [0.3, 0.5]

    n_scorer = CandidateSet(articles=article_list_1, scores=score_list_1)
    n_scorer2 = CandidateSet(articles=article_list_2, scores=score_list_2)

    fusion = ScoreFusion()

    output_1 = fusion(candidates1=n_scorer, candidates2=n_scorer2)

    assert output_1.scores == [0.2, 0.6, 0.5]


def test_empty_overlapped_articleset():
    article1 = Article(headline="A")
    article2 = Article(headline="B")

    article_list_1 = [article1, article2]
    score_list_1 = [0.2, 0.3]

    n_scorer0 = CandidateSet(articles=[], scores=[])
    n_scorer = CandidateSet(articles=article_list_1, scores=score_list_1)

    fusion = ScoreFusion()
    output_01 = fusion(candidates1=n_scorer0, candidates2=n_scorer)

    assert output_01.scores == [0.2, 0.3]


def test_overlapped_articleset_avg():
    article1 = Article(headline="A")
    article2 = Article(headline="B")
    article3 = Article(headline="C")

    article_list_1 = [article1, article2]
    score_list_1 = [0.2, 0.3]

    article_list_2 = [article2, article3]
    score_list_2 = [0.3, 0.5]

    n_scorer = CandidateSet(articles=article_list_1, scores=score_list_1)
    n_scorer2 = CandidateSet(articles=article_list_2, scores=score_list_2)

    config = ScoreFusionConfig(combiner="avg")
    fusion = ScoreFusion(config)

    output_1 = fusion(candidates1=n_scorer, candidates2=n_scorer2)

    assert output_1.scores == [0.1, 0.3, 0.25]


def test_positive_negative_score():
    article1 = Article(headline="A")
    article2 = Article(headline="B")
    article3 = Article(headline="C")

    article_list_1 = [article1, article2]
    score_list_1 = [0.2, 0.3]

    article_list_2 = [article2, article3]
    score_list_2 = [0.3, 0.5]

    n_scorer = CandidateSet(articles=article_list_1, scores=score_list_1)
    n_scorer2 = CandidateSet(articles=article_list_2, scores=score_list_2)

    config = ScoreFusionConfig(combiner="sub")
    fusion = ScoreFusion(config)

    output_1 = fusion(candidates1=n_scorer, candidates2=n_scorer2)

    assert output_1.scores == [0.2, 0, -0.5]
