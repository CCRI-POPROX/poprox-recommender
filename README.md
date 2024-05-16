# poprox-recommender

This repository contains the POPROX recommender code — the end-to-end logic for
producing recommendations using article data, user histories and profiles, and
trained models.

- [Installation](#installation)
- [Local Development](#localdevelopment)
- [License](#license)

## Installation

Model and data files are managed using [dvc][].  The `conda-lock.yml` provides a
[conda-lock][] lockfile for reproducibly creating an environment with all
necessary dependencies.

[dvc]: https://dvc.org
[conda-lock]: https://conda.github.io/conda-lock/

To set up the environment with Conda:

```
conda install -n base -c conda-forge conda-lock
conda lock install -n poprox-recsys
conda activate poprox-recsys
```

If you use `micromamba` instead of a full Conda installation, it can directly use the lockfile:

```
micromamba create -n poprox-recs -f conda-lock.yml
```

To get the data and models, there are two steps:

1.  Obtain the credentials for the S3 bucket and put them in `.env` (the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
2.  `dvc pull`

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

Also, you can test the rec model with MIND Large test dataset by running the `eval_with_mind_large_test.py` file, which will produce
AUC, MRR, nDCG@5, nDCG@10 values.

To add large data or model file via dvc:

1. install dvc dependencies if you don't have them already

    `pip install dvc`
    `pip install dvc_s3`

2. Obtain the credentials for the S3 bucket and put them in `.env` (the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`).
If this step does not work, you can also try to directly export them with 

    `export AWS_ACCESS_KEY_ID="REPLACE_WITH_KEY_IDENTIFIERS"`
    
    `export AWS_SECRET_ACCESS_KEY="REPLACE_WITH_SECRET_ACCESS_KEY"`


3. Add the data or model files you wish to push to the repo:

    `dvc add <filename>`
    `dvc push`

4. Use git to update the new dvc file and commit and push as usual:

    `git add <filename.dvc>`
    `git commit -m "YOUR_COMMIT_MESSAGE"`
    `git push`
