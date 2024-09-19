# poprox-recommender

This repository contains the POPROX recommender code — the end-to-end logic for
producing recommendations using article data, user histories and profiles, and
trained models.

- [Installation](#installation)
- [Local Development](#localdevelopment)
- [Editor Setup](#editor-setup)
- [License](#license)

## Installation for Development

Software environments for this repository are managed with [pixi][], and model
and data files are managed using [dvc][]. The `pixi.lock` file provides a locked
dependency set for reproducibly running the recommender code with all
dependencies, on Linux, macOS, and Windows (including with CUDA on Linux).

[pixi]: https://pixi.sh
[dvc]: https://dvc.org

To install the dependencies needed for development work:

```console
pixi install -e dev
```

Alternatively, on Linux, you can use `cuda` instead of `dev`.

Once you have installed the dependencies, there are 2 easy ways to run code in the environment:

1.  Run individual commands with `pixi run`, e.g.:

    ```console
    pixi run -e dev pytest tests
    ```

2.  Run a Pixi shell, which activates the environment and adds the appropriate
    Python to your `PATH`:

    ```console
    pixi shell -e dev
    ```

> [!NOTE]
> Tests are also available as a Pixi task: `pixi run -e dev test`.


Finally, set up `pre-commit` to make sure that code formatting rules are applied
as you make changes:

```console
pre-commit install
```

> [!NOTE]
>
> If you will be working with `git` outside of the `pixi` shell, you may want to
> install `pre-commit` separately.  You can do this with Brew or your preferred
> system or user package manager, or with `pixi global install pre-commit`.

To get the data and models, there are two steps:

1.  Obtain the credentials for the S3 bucket and put them in `.env` (the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
2.  `dvc pull`

### Dependency Updates

If you update the dependencies in poprox-recommender, or add code that requires
a newer version of `poprox-concepts`, you need to regenerate the lock file with
`pixi update`.  To update just `poprox-concepts`, run:

```console
pixi update poprox_concepts
```

To update all dependencies, run:

```console
pixi update
```

> [!NOTE]
> Currently, dependencies can only be updated on Linux.

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
      "headline": "headline 1",
      "subhead": "subhead 1",
      "url": "url 1"
    }
  ],
  "todays_articles": [
    {
      "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
      "headline": "headline 2",
      "subhead": "subhead 2",
      "url": "url 2"
    }
  ],
  "interest_profile": {
    "profile_id": "28838f05-23f5-4f23-bea2-30b51f67c538",
    "click_history": [
      {
        "article_id": "e7605f12-a37a-4326-bf3c-3f9b72d0738d"
      }
    ],
    "onboarding_topics": []
  },
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
$ curl -X POST -H "Content-Type: application/json" -d @tests/request_data/basic-request.json localhost:3000

{"recommendations": {"977a3c88-937a-46fb-bbfe-94dc5dcb68c8": [{"article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff", "title": "title 2", "content": "content 2", "url": "url 2", "published_at": "1970-01-01T00:00:00Z", "mentions": []}]}}
```

## Running the Evaluation

The default setup for this package is CPU-only, which works for basic testing
and for deployment, but is very inefficient for evaluation.  The current set of
models work on both CUDA (on Linux with NVidia cards) and MPS (macOS on Apple
Silicon).  To make use of a GPU, do the following:

1.  If on Linux, install the CUDA-based Conda environment:

    ```console
    pixi install -e cuda
    ```

2.  Set the `POPROX_REC_DEVICE` environment variable to `cuda` or `mps`.

3.  Run `dvc repro` under the `cuda` environment (using either `pixi run` or
    `pixi shell`).

Timing information for generating recommendations with the MIND validation set:

| CPU              | GPU        | Rec. Time | Eval Time |
| :--------------: | :--------: | :-------: | :-------: |
| EPYC 7662 (2GHz) | A40 (CUDA) | 2h10m     | 45m       |
| Apple M2 Pro     | -          | <20hr¹    | 30m¹      |
| Apple M2 Pro     | M2 (MPS)   | <12hr¹    |           |

Footnotes:

1. Estimated based on early progress, not run to completion.

## Editor Setup

If you are using VSCode, you should install the following plugins for best success with this repository:

- [EditorConfig](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

When you open the repository, they should automatically be recommended.
