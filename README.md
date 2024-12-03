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

See the Pixi [install instructions][pixi] for how to install Pixi in general. On
macOS, you can also use Homebrew (`brew install pixi`), and on Windows you can
use WinGet (`winget install prefix-dev.pixi`).

> [!NOTE]
>
> If you are trying to work on poprox-recommender with WSL on Windows, you need
> to follow the Linux install instructions, and also add the following to the
> Pixi configuration file (`~/.pixi/config.toml`):
>
> ```toml
> detached-environments = true
> ```

[pixi]: https://pixi.sh
[dvc]: https://dvc.org

Once Pixi is installed, to install the dependencies needed for development work:

```console
pixi install -e dev
```

Once you have installed the dependencies, there are 3 easy ways to run code in the environment:

1.  Run a defined task, like `test`, with `pixi run`:

    ```console
    pixi run -e dev test
    ```

2.  Run individual commands with `pixi run`, e.g.:

    ```console
    pixi run -e dev pytest tests
    ```

3.  Run a Pixi shell, which activates the environment and adds the appropriate
    Python to your `PATH`:

    ```console
    pixi shell -e dev
    ```

> [!NOTE]
>
> If you have a CUDA-enabled Linux system, you can use the `dev-cuda` and
> `eval-cuda` environments to use your GPU for POPROX batch inference.

> [!NOTE]
>
> `pixi shell` starts a new, *nested* shell with the Pixi environment active. If
> you type `exit` in this shell, it will exit the nested shell and return you to
> you original shell session without the environment active.

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
> We use [Pixi][] for all dependency management.  If you need to add a new dependency
> for this code, add it to the appropriate feature(s) in `pixi.toml`.  If it is a
> dependency of the recommendation components themselves, add it both to the
> top-level `dependencies` table in `pixi.toml` *and* in `pyproject.toml`.

> [!NOTE]
> Currently, dependencies can only be updated on Linux.

## Local Endpoint Development

Local endpoint development also requires some Node-based tools in addition to the tools above — this install is
automated with `npm` and `pixi`:

```console
pixi run -e dev install-serverless
```

To run the API endpoint locally:

```console
pixi run -e dev start-serverless
```

<details>
<summary>Implementation</sumamry>

Under the hood, those tasks run the following Node commands within the `dev` environment:

```console
npm ci
npx serverless offline start --reloadHandler
```
</details>

Once the local server is running, you can send requests to `localhost:3000`. A request with this JSON body:

```json
{
  "past_articles": [
    {
      "article_id": "e7605f12-a37a-4326-bf3c-3f9b72d0738d",
      "headline": "headline 1",
      "subhead": "subhead 1",
      "url": "url 1"
    },
    {
      "article_id": "a0266b75-2873-4a40-9373-4e216e88c2f7",
      "headline": "headline 2",
      "subhead": "subhead 2",
      "url": "url 2"
    },
    {
      "article_id": "d88ee3b6-2b5e-4821-98a8-ffd702f571de",
      "headline": "headline 3",
      "subhead": "subhead 3",
      "url": "url 3"
    }
  ],
  "todays_articles": [
    {
      "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
      "headline": "headline 4",
      "subhead": "subhead 4",
      "url": "url 4"
    },
    {
      "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
      "headline": "headline 5",
      "subhead": "subhead 5",
      "url": "url 5"
    },
    {
      "article_id": "2d5a25ba-0963-474a-8da6-5b312c87bb82",
      "headline": "headline 6",
      "subhead": "subhead 6",
      "url": "url 6"
    }
  ],
  "interest_profile": {
    "profile_id": "28838f05-23f5-4f23-bea2-30b51f67c538",
    "click_history": [
      {
        "article_id": "e7605f12-a37a-4326-bf3c-3f9b72d0738d"
      },
      {
        "article_id": "a0266b75-2873-4a40-9373-4e216e88c2f7"
      },
      {
        "article_id": "d88ee3b6-2b5e-4821-98a8-ffd702f571de"
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
        "28838f05-23f5-4f23-bea2-30b51f67c538": [
            {
                "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
                "headline": "headline 5",
                "subhead": "subhead 5",
                "url": "url 5",
                "preview_image_id": null,
                "published_at": "1970-01-01T00:00:00Z",
                "mentions": [],
                "source": null,
                "external_id": null,
                "raw_data": null
            },
            ...
        ]
    },
    "recommender": {
        "name": "plain-NRMS",
        "version": null,
        "hash": "bd076520fa51dc70d8f74dfc9c0e0169236478d342b08f35b95399437f012563"
    }
}
```

You can test this by sending a request with curl:

```console
$ curl -X POST -H "Content-Type: application/json" -d @tests/request_data/basic-request.json localhost:3000
```

## Running the Evaluation

The default setup for this package is CPU-only, which works for basic testing
and for deployment, but is very inefficient for evaluation.  The current set of
models work on both CUDA (on Linux with NVidia cards) and MPS (macOS on Apple
Silicon).  To make use of a GPU, do the following:, run `dvc repro` under the
`eval-cuda` or `dev-cuda` environment (using either `pixi run` or `pixi shell`).

Timing information for generating recommendations with the MIND validation set:

| Machine | CPU                | GPU         | Rec. Time | Rec. Power | Eval Time |
| ------- | :----------------: | :---------: | :-------: | :--------: | :-------: |
| [DXC][] | EPYC 7662 (2GHz)   | A40 (CUDA)  | 45m¹      | 418.5 Wh   | 24m       |
| [DXS][] | i9 14900K (3.2GHz) | 4090 (CUDA) | 30m²      |            | 14m       |
| [MBP][] | Apple M2 Pro       | -           | <20hr³    |            | 30m³      |
| [MBP][] | Apple M2 Pro       | M2 (MPS)    | <12hr³    |            |           |

[DXC]: https://codex.lenskit.org/hardware/cruncher.html
[DXS]: https://codex.lenskit.org/hardware/screamer.html
[MBP]: https://codex.lenskit.org/hardware/ranger.html

Footnotes:

1. Using 12 worker processes
2. Using 8 worker processes
3. Estimated based on early progress, not run to completion.

## Editor Setup

If you are using VSCode, you should install the following plugins for best success with this repository:

- [EditorConfig](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

When you open the repository, they should be automatically recommended.
