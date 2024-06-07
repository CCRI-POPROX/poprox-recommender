import json

from poprox_concepts import Article, ClickHistory
from poprox_recommender.default import select_articles


def generate_recs(event, context):
    req_body = json.loads(event["body"])

    todays_articles = [
        Article.model_validate(attrs) for attrs in req_body["todays_articles"]
    ]
    past_articles = [
        Article.model_validate(attrs) for attrs in req_body["past_articles"]
    ]
    click_data = [
        ClickHistory.model_validate(attrs) for attrs in req_body["click_data"]
    ]
    num_recs = req_body["num_recs"]

    recommendations = select_articles(
        todays_articles,
        past_articles,
        click_data,
        num_recs,
    )

    body = {
        "recommendations": {
            str(account_id): [
                json.loads(article.model_dump_json()) for article in articles
            ]
            for account_id, articles in recommendations.items()
        },
    }

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response
