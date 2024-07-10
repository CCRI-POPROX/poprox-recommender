# poprox-recommender

This repository contains the POPROX recommender code — the end-to-end logic for
producing recommendations using article data, user histories and profiles, and
trained models.

- [Installation](#installation)
- [Local Development](#localdevelopment)
- [Editor Setup](#editor-setup)
- [License](#license)

## Installation for Development

Model and data files are managed using [dvc][].  The `conda-lock.yml` provides a
[conda-lock][] lockfile for reproducibly creating an environment with all
necessary dependencies.

[dvc]: https://dvc.org
[conda-lock]: https://conda.github.io/conda-lock/

To set up the environment with Conda:

```console
conda install -n base -c conda-forge conda-lock
conda lock install -n poprox-recsys --dev
conda activate poprox-recsys
python -m pip install --no-deps -e .
```

If you use `micromamba` instead of a full Conda installation, it can directly use the lockfile:

```console
micromamba create -n poprox-recs -f conda-lock.yml --category main --category dev
python -m pip install --no-deps -e .
```

> [!NOTE]
> You need to re-run the `pip install` every time you re-create your Conda environment.

Set up `pre-commit` to make sure that code formatting rules are applied as you make changes:

```console
pre-commit install
```

To get the data and models, there are two steps:

1.  Obtain the credentials for the S3 bucket and put them in `.env` (the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
2.  `dvc pull`

### Dependency Updates

If you update the dependencies in poprox-recommender, or add code that requires
a newer version of `poprox-concepts`, you need to regenerate the lock file:

```console
./dev/update-dep-lock.sh
```

> [!NOTE]
> If you are trying to re-lock the dependencies on Windows, you should run the
> shell script from within a Git Bash or MSYS2 environment.

## Local Endpoint Development

Local endpoint development also requires some Node-based tools in addition to the tools above:

```console
npm install
```

To run the API endpoint locally:

```console
npx serverless offline start --reloadHandler
```

Once the local server is running, you can send requests to `localhost:3000`. A request with this JSON body:

```json
{
    "past_articles": [
        {
            "article_id": "e7605f12-a37a-4326-bf3c-3f9b72d0738d",
            "title": "title 1",
            "content": "content 1",
            "url": "url 1"
        }
    ],
    "todays_articles": [
        {
            "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
            "title": "title 2",
            "content": "content 2",
            "url": "url 2"
        }
    ],
    "click_histories": [
        {
            "account_id": "977a3c88-937a-46fb-bbfe-94dc5dcb68c8",
            "article_ids": [
                "e7605f12-a37a-4326-bf3c-3f9b72d0738d"
            ]
        }
    ],
    "num_recs": 1
}
```

should receive this response:

```json
{
    "recommendations": {
        "977a3c88-937a-46fb-bbfe-94dc5dcb68c8": [
            {
                "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
                "title": "title 2",
                "content": "content 2",
                "url": "url 2",
                "published_at": "1970-01-01T00:00:00Z",
                "mentions": []
            }
        ]
    }
}
```

You can test this by sending a request with curl:

```console
$ curl -X POST -H "Content-Type: application/json" -d @tests/basic-request.json localhost:3000

{"recommendations": {"977a3c88-937a-46fb-bbfe-94dc5dcb68c8": [{"article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff", "title": "title 2", "content": "content 2", "url": "url 2", "published_at": "1970-01-01T00:00:00Z", "mentions": []}]}}
```

## Running the Evaluation

The default setup for this package is CPU-only, which works for basic testing
and for deployment, but is very inefficient for evaluation.  The current set of
models work on both CUDA (on Linux with NVidia cards) and MPS (macOS on Apple
Silicon).  To make use of a GPU, do the following:

1.  If on Linux, install the CUDA-based Conda environment:

    ```console
    conda-lock install -n poprox-recs --dev conda-lock-cuda.yml
    ```

2.  Set the `POPROX_REC_DEVICE` environment variable to `cuda` or `mps`.

Timing information for batch evaluation with the MIND validation set:

| CPU              | GPU        | Time   | Notes                            |
| :--------------: | :--------: | :----: | -------------------------------- |
| EPYC 7662 (2GHz) | A40 (CUDA) | <1hr   |                                  |
| Apple M2 Pro     | M2 (MPS)   | ~1 day | Estimated, not run to completion |
| Apple M2 Pro     | -          | days   | Estimated, not run to completion |

## Editor Setup

If you are using VSCode, you should install the following plugins for best success with this repository:

- [EditorConfig](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

When you open the repository, they should automatically be recommended.
