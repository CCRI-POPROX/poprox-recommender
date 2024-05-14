# poprox-recommender

[![PyPI - Version](https://img.shields.io/pypi/v/poprox-recommender.svg)](https://pypi.org/project/poprox-recommender)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/poprox-recommender.svg)](https://pypi.org/project/poprox-recommender)

-----

**Table of Contents**

- [Installation](#installation)
- [Local Development](#localdevelopment)
- [License](#license)

## Installation

```console
pip install poprox-recommender
```

## Local Development

There are two sets of dependencies. To install the Serverless framework and Node dependencies:

```console
npm install -g serverless
npm install
```

To install Python dependencies:

```console
pip install -r requirements.txt
```

To run the API endpoint locally:

```console
serverless offline start --reloadHandler
```

Once the local server is running, you can send requests to `localhost:3000`. A request with this JSON body:

```json
{
    "past_articles": [
                {
            "article_id": "1",
            "title": "title 1",
            "content": "content 1",
            "url": "url 1"
        }
    ],
    "todays_articles": [
        {
            "article_id": "2",
            "title": "title 2",
            "content": "content 2",
            "url": "url 2"
        }
    ],
    "click_data": {"user 1": ["url 1"]},
    "num_recs": 1
}
```

should receive this response:

```json
{
    "recommendations": {
        "user 1": [
            {
                "article_id": "2",
                "title": "title 2",
                "content": "content 2",
                "url": "url 2"
            }
        ]
    }
}
```

You can test this by sending a request with curl:

```console
$ curl -X POST -H "Content-Type: application/json" -d @tests/basic-request.json localhost:3000

{"recommendations": {"user 1": [{"article_id": "2", "title": "title 2", "content": "content 2", "url": "url 2"}]}}
```
